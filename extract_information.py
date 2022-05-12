import os
from os import listdir
from os.path import isfile, join
import dlib
import imgsim
import glob
import pickle
import cv2
import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch.nn import functional as F
import PIL
from PIL import Image
import numpy as np
import csv
from google_trans_new import google_translator
import pyopenpose as op
import math
from operator import add
from sentence_transformers import SentenceTransformer
from iptcinfo3 import IPTCInfo
from pathlib import Path
from omegaconf import OmegaConf
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from torch import nn
from torchvision import models
from torchvision import transforms
import torch.nn.init as init
import json
from imageio import imread
import clip
from typing import Tuple, List, Union, Optional
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
import skimage.io as io
from skimage.exposure import is_low_contrast

features_blobs = []

def extract_face_information(images_folder):
    predictor_path = 'shape_predictor_5_face_landmarks.dat'
    face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
    faces_folder_path = images_folder

    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    face_dict = {"Filename": [], "Rectangles": [], "Descriptors": []};
    file_counter = 1
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):

        img = dlib.load_rgb_image(f)

        if img.shape[0] > 1500:
            scale_percent = (1500 / img.shape[0]) * 100  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        dets = detector(img, 1)

        face_dict["Filename"].append(f.split('\\')[1])
        face_dict["Rectangles"].append(dets)

        descriptors = []
        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            descriptors.append(face_descriptor)
        face_dict["Descriptors"].append(descriptors)
        if file_counter%10==0: print(str(file_counter) + ' / 23502')
        file_counter += 1

    with open('data_pickle/' + images_folder_name + '/faces.pickle', 'wb') as handle:
        pickle.dump(face_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rows = zip(face_dict['Filename'], face_dict['Rectangles'], face_dict['Descriptors'])
    with open('data_csv/' + images_folder_name + '/faces.csv', "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(face_dict.keys())
        for row in rows:
            writer.writerow(row)

    return

def extract_image_vectors(images_folder):
    base_folder = images_folder + '/'

    vtr = imgsim.Vectorizer()

    file_list = [f for f in listdir(base_folder) if isfile(join(base_folder, f))]

    vectors_dict = dict()
    file_counter = 1
    for filename in file_list:
        img1 = cv2.imread(base_folder + filename)

        vec1 = vtr.vectorize(img1)
        vectors_dict.update({filename: vec1})
        if file_counter%10==0: print(str(file_counter) + ' / 23502')
        file_counter += 1

    with open('data_pickle/' + images_folder_name + '/vectors.pickle', 'wb') as handle:
        pickle.dump(vectors_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data_csv/' + images_folder_name + '/vectors.csv', "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Vector'])
        for key, value in vectors_dict.items():
            writer.writerow([key, value])

    return

def hook_feature(module, input, output):
    global features_blobs
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def extract_scene_information(images_folder):
    environment_dict = {"Filename": [], "IO": [], "Environments": []};
    global features_blobs
    arch = 'resnet18'
    model_file = '%s_places365.pth.tar' % arch
    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    features_names = ['layer4', 'avgpool']  # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    model.load_state_dict(state_dict)
    model.eval()
    centre_crop = trn.Compose([
        trn.Resize((256, 256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    file_name = 'categories_places365.txt'
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)
    file_name_IO = 'IO_places365.txt'
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) - 1)  # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)
    base_folder = images_folder + '/'
    file_list = [f for f in listdir(base_folder) if isfile(join(base_folder, f))]
    file_counter = 1
    for file in file_list:

        filePath = base_folder + file

        img = PIL.Image.open(filePath)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        input_img = V(centre_crop(img).unsqueeze(0))
        input_img = torch.from_numpy(input_img).float()
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        io_image = np.mean(labels_IO[idx[:10]])  # vote for the indoor or outdoor
        if io_image < 0.5:
            indoor_outdoor = 'indoor'
        else:
            indoor_outdoor = 'outdoor'

        environment_dict["Filename"].append(file)
        environment_dict["IO"].append(indoor_outdoor)

        environment_info_list = []
        for i in range(0, 5):
            environment_info_list.append(str(probs[i])[7:][:-1] + ' ' + str(classes[idx[i]]))
        environment_dict["Environments"].append(environment_info_list)

        if file_counter%10==0: print(str(file_counter) + ' / 23502')
        file_counter += 1

    with open('data_pickle/' + images_folder_name + '/environments.pickle', 'wb') as handle:
        pickle.dump(environment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rows = zip(environment_dict['Filename'], environment_dict['IO'], environment_dict['Environments'])
    with open('data_csv/' + images_folder_name + '/environments.csv', "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(environment_dict.keys())
        for row in rows:
            writer.writerow(row)

    return

def extract_object_information(images_folder):
    object_dict = {"Filename": [], "Objects": [], "Rectangles": []};

    net = cv2.dnn_DetectionModel('yolov4.cfg.txt', 'yolov4.weights')
    net.setInputSize(704, 704)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)

    base_folder = images_folder + '/'

    file_list = [f for f in listdir(base_folder) if isfile(join(base_folder, f))]
    with open('coco.names.txt', 'rt') as f:
        names = f.read().rstrip('\n').split('\n')
    file_counter = 1
    for filename in file_list:

        frame = cv2.imread(base_folder + filename)

        image_resized = False
        if frame.shape[0] > 750:
            scale_percent = (750 / frame.shape[0]) * 100  # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            image_resized = True

        object_dict["Filename"].append(filename)
        object_info_list = []
        rectangles_list = []
        classes, confidences, boxes = net.detect(frame, confThreshold=0.25, nmsThreshold=0.4)
        if len(classes) > 0:
            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                object_info_list.append(str(confidence) + ' ' + names[classId])
                #left, top, width, height = box
                if image_resized:
                    box = box * 100 / scale_percent
                    box = box.astype(int)
                rectangles_list.append(box)
        object_dict["Objects"].append(object_info_list)
        object_dict["Rectangles"].append(rectangles_list)
        if file_counter%10==0: print(str(file_counter) + ' / 23502')
        file_counter += 1

    with open('data_pickle/' + images_folder_name + '/objects.pickle', 'wb') as handle:
        pickle.dump(object_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rows = zip(object_dict['Filename'], object_dict['Objects'], object_dict['Rectangles'])
    with open('data_csv/' + images_folder_name + '/objects.csv', "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(object_dict.keys())
        for row in rows:
            writer.writerow(row)

    return


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = np.array(PIL.Image.fromarray(img).resize((256, 256)))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds.long()], alpha[prev_word_inds.long()].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds.long()[incomplete_inds]]
        c = c[prev_word_inds.long()[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds.long()[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas

def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = PIL.Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], PIL.Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]
    sentence = ''
    for word in words:
        if word != '<start>' and word != '<end>':
            sentence += word + ' '
    sentence = sentence[:-1]
    sentence += '.'

    return sentence

def generate_beam(
        model,
        tokenizer,
        beam_size: int = 5,
        prompt=None,
        embed=None,
        entry_length=67,
        temperature=1.0,
        stop_token: str = ".",
):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

D = torch.device
CPU = torch.device("cpu")

class MLP(nn.Module):
    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    # @functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(
            self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None
    ):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        # print(embedding_text.size()) #torch.Size([5, 67, 768])
        # print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(
                prefix_size, self.gpt_embedding_size * prefix_length
            )
        else:
            self.clip_project = MLP(
                (
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                )
            )


class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def extract_image_captions(images_folder):
    sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')

    caption_dict = {"Filename": [], "English1": [], "English2": [], "English3": [], "Finnish1": [], "Finnish2": [], "Finnish3": [],  "Embeddings1": [], "Embeddings2": [], "Embeddings3": []};

    base_folder = images_folder + '/'

    file_list = [f for f in listdir(base_folder) if isfile(join(base_folder, f))]

    translator = google_translator()

    language = "en"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = 'BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    checkpoint = torch.load(model1, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    word_map_file = 'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'

    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    beam_size = 5

    smooth = True

    WEIGHTS_PATHS = {
        "coco": "coco_weights.pt",
        "conceptual-captions": "conceptual_weights.pt",
    }

    device = torch.device("cpu")
    clip_model, preprocess = clip.load(
        "ViT-B/32", device=device, jit=False
    )
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    models = {}
    prefix_length = 10
    for key, weights_path in WEIGHTS_PATHS.items():
        model = ClipCaptionModel(prefix_length)
        model.load_state_dict(torch.load(weights_path, map_location=CPU))
        model = model.eval()
        model = model.to(device)
        models[key] = model

    modelCOCO = models['coco']
    modelCC = models['conceptual-captions']
    file_counter = 1
    for filename in file_list:
        seq, alphas = caption_image_beam_search(encoder, decoder, base_folder + filename, word_map, beam_size)
        alphas = torch.FloatTensor(alphas)

        # Visualize caption and attention of best sequence
        sentence = visualize_att(base_folder + filename, seq, alphas, rev_word_map, smooth)

        caption_dict["Filename"].append(filename)

        caption_dict["English1"].append(sentence)
        #caption_dict["Finnish1"].append(translator.translate(sentence, lang_tgt='fi'))
        caption_dict["Finnish1"].append('')
        caption_dict["Embeddings1"].append(sentence_model.encode(sentence))

        image = io.imread(base_folder + filename)
        pil_image = PIL.Image.fromarray(image)
        image = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            prefix = clip_model.encode_image(image).to(
                device, dtype=torch.float32
            )
            prefix_embedCOCO = modelCOCO.clip_project(prefix).reshape(1, prefix_length, -1)
            prefix_embedCC = modelCC.clip_project(prefix).reshape(1, prefix_length, -1)

        resultCOCO = generate_beam(modelCOCO, tokenizer, embed=prefix_embedCOCO)[0]
        resultCC = generate_beam(modelCC, tokenizer, embed=prefix_embedCC)[0]

        #caption_dict["English2"].append('')
        caption_dict["English2"].append(resultCOCO)
        #caption_dict["Finnish2"].append(translator.translate(resultCOCO, lang_tgt='fi'))
        caption_dict["Finnish2"].append('')
        caption_dict["Embeddings2"].append(sentence_model.encode(resultCOCO))
        #caption_dict["Embeddings2"].append('')

        #caption_dict["English3"].append('')
        caption_dict["English3"].append(resultCC)
        #caption_dict["Finnish3"].append(translator.translate(resultCC, lang_tgt='fi'))
        caption_dict["Finnish3"].append('')
        caption_dict["Embeddings3"].append(sentence_model.encode(resultCC))
        #caption_dict["Embeddings3"].append('')

        if file_counter%10==0: print(str(file_counter) + ' / 23502')
        file_counter += 1

    with open('data_pickle/' + images_folder_name + '/captions2.pickle', 'wb') as handle:
        pickle.dump(caption_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rows = zip(caption_dict['Filename'], caption_dict['English1'], caption_dict['English2'], caption_dict['English3'], caption_dict['Finnish1'], caption_dict['Finnish2'], caption_dict['Finnish3'], caption_dict['Embeddings1'], caption_dict['Embeddings2'], caption_dict['Embeddings3'])
    with open('data_csv/' + images_folder_name + '/captions2.csv', "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(caption_dict.keys())
        for row in rows:
            writer.writerow(row)

    return

def getBlurValue(image):
    canny = cv2.Canny(image, 50, 250)
    return np.mean(canny)

def gaze(face_keypoints, full_body_keypoints):
    # perspective of the photo viewer
    leftEar = face_keypoints[0][2] > 0
    leftEye = face_keypoints[1][2] > 0
    nose = face_keypoints[2][2] > 0
    rightEye = face_keypoints[3][2] > 0
    rightEar = face_keypoints[4][2] > 0

    facialKeypointConfidenceSum = 0
    if leftEar: facialKeypointConfidenceSum += face_keypoints[0][2]
    if leftEye: facialKeypointConfidenceSum += face_keypoints[1][2]
    if nose: facialKeypointConfidenceSum += face_keypoints[2][2]
    if rightEye: facialKeypointConfidenceSum += face_keypoints[3][2]
    if rightEar: facialKeypointConfidenceSum += face_keypoints[4][2]
    length = int(leftEar) + int(leftEye) + int(nose) + int(rightEye) + int(rightEar)
    facialConfidenceAverage = facialKeypointConfidenceSum / length

    leftShoulderX = full_body_keypoints[2][0]
    rightShoulderX = full_body_keypoints[5][0]

    leftEyeX = face_keypoints[1][0]
    rightEyeX = face_keypoints[3][0]
    leftEarX = face_keypoints[0][0]
    rightEarX = face_keypoints[4][0]
    eyeDistance = rightEyeX - leftEyeX
    earDistance = rightEarX - leftEarX

    direction = 'undefined'

    if leftEye and rightEye and nose:
        direction = 'direct'
    if not leftEye and rightEye:
        direction = 'left'
    if leftEye and not rightEye:
        direction = 'right'
    if not leftEye and not rightEye:
        direction = 'away'
    if not nose:
        direction = 'away'
    if facialConfidenceAverage < 0.45:
        direction = 'undefined'
    if leftEar and leftEye and rightEar and rightEye and earDistance / eyeDistance > 50:
        direction = 'undefined'
    if leftEyeX > rightEyeX and leftEye and rightEye:
        direction = 'undefined'

    return direction

def face_rectangles(keypoints, image_width, image_height):
    rectangles = []
    gazes = []
    associated_keypoints = []
    for keypoint in keypoints:
        facial_keypoints = [keypoint[17], keypoint[15], keypoint[0], keypoint[16], keypoint[18]]

        x_locations = []
        y_locations = []
        for facial_keypoint in facial_keypoints:
            confidence = facial_keypoint[2]
            if confidence > 0:
                x_locations.append(facial_keypoint[0])
                y_locations.append(facial_keypoint[1])
        if len(x_locations) == 0 or len(y_locations) == 0:
            continue
        min_x = min(x_locations)
        max_x = max(x_locations)
        for facial_keypoint in facial_keypoints:
            if facial_keypoint[0] == min_x:
                leftmost_point = (facial_keypoint[0], facial_keypoint[1])
            if facial_keypoint[0] == max_x:
                rightmost_point = (facial_keypoint[0], facial_keypoint[1])
        if len(x_locations) >= 2 and len(y_locations) >= 2:
            width = max_x - min_x
            midpoint = ((leftmost_point[0] + rightmost_point[0]) / 2, (leftmost_point[1] + rightmost_point[1]) / 2)
            top_left = (midpoint[0] - width / 2, midpoint[1] - width / 2)
            bottom_right = (midpoint[0] + width / 2, midpoint[1] + width / 2)
            if top_left[0] < 0: top_left = (0, top_left[1])
            if top_left[1] < 0: top_left = (top_left[0], 0)
            if bottom_right[0] >= image_width: bottom_right = (image_width - 1, bottom_right[1])
            if bottom_right[1] >= image_height: bottom_right = (bottom_right[0], image_height - 1)
            rectangle = [top_left, bottom_right]
            rectangles.append(rectangle)
            associated_keypoints.append(keypoint)

            facial_keypoints = [keypoint[17], keypoint[15], keypoint[0], keypoint[16], keypoint[18]]
            gaze_direction = gaze(facial_keypoints, keypoint)
            gazes.append(gaze_direction)

        else:
            continue

    return rectangles, gazes, associated_keypoints

def gaze_2(face_keypoints, full_body_keypoints):
    # perspective of the photo viewer
    leftEar = face_keypoints[0][2] > 0
    leftEye = face_keypoints[1][2] > 0
    nose = face_keypoints[2][2] > 0
    rightEye = face_keypoints[3][2] > 0
    rightEar = face_keypoints[4][2] > 0

    facialKeypointConfidenceSum = 0
    if leftEar: facialKeypointConfidenceSum += face_keypoints[0][2]
    if leftEye: facialKeypointConfidenceSum += face_keypoints[1][2]
    if nose: facialKeypointConfidenceSum += face_keypoints[2][2]
    if rightEye: facialKeypointConfidenceSum += face_keypoints[3][2]
    if rightEar: facialKeypointConfidenceSum += face_keypoints[4][2]
    length = int(leftEar) + int(leftEye) + int(nose) + int(rightEye) + int(rightEar)
    facialConfidenceAverage = facialKeypointConfidenceSum / length

    leftShoulderX = full_body_keypoints[2][0]
    rightShoulderX = full_body_keypoints[5][0]

    leftEyeX = face_keypoints[1][0]
    rightEyeX = face_keypoints[3][0]
    leftEarX = face_keypoints[0][0]
    rightEarX = face_keypoints[4][0]
    eyeDistance = rightEyeX - leftEyeX
    earDistance = rightEarX - leftEarX

    direction = 'undefined'

    if leftEye and rightEye and nose and leftEar and rightEar:
        direction = 'direct'
    if not leftEar and rightEar and nose:
        direction = 'left'
    if leftEar and not rightEar and nose:
        direction = 'right'
    if not leftEye and not rightEye:
        direction = 'away'
    if not nose:
        direction = 'away'
    if facialConfidenceAverage < 0.45:
        direction = 'undefined'
    if leftEar and leftEye and rightEar and rightEye and earDistance / eyeDistance > 50:
        direction = 'undefined'
    if leftEyeX > rightEyeX and leftEye and rightEye:
        direction = 'undefined'

    return direction

def face_rectangles_2(keypoints, image_width, image_height):
    rectangles = []
    gazes = []
    associated_keypoints = []
    for keypoint in keypoints:
        facial_keypoints = [keypoint[17], keypoint[15], keypoint[0], keypoint[16], keypoint[18]]

        x_locations = []
        y_locations = []
        for facial_keypoint in facial_keypoints:
            confidence = facial_keypoint[2]
            if confidence > 0:
                x_locations.append(facial_keypoint[0])
                y_locations.append(facial_keypoint[1])
        if len(x_locations) == 0 or len(y_locations) == 0:
            continue
        min_x = min(x_locations)
        max_x = max(x_locations)
        for facial_keypoint in facial_keypoints:
            if facial_keypoint[0] == min_x:
                leftmost_point = (facial_keypoint[0], facial_keypoint[1])
            if facial_keypoint[0] == max_x:
                rightmost_point = (facial_keypoint[0], facial_keypoint[1])
        if len(x_locations) >= 2 and len(y_locations) >= 2:
            width = max_x - min_x
            midpoint = ((leftmost_point[0] + rightmost_point[0]) / 2, (leftmost_point[1] + rightmost_point[1]) / 2)
            top_left = (midpoint[0] - width / 2, midpoint[1] - width / 2)
            bottom_right = (midpoint[0] + width / 2, midpoint[1] + width / 2)
            if top_left[0] < 0: top_left = (0, top_left[1])
            if top_left[1] < 0: top_left = (top_left[0], 0)
            if bottom_right[0] >= image_width: bottom_right = (image_width - 1, bottom_right[1])
            if bottom_right[1] >= image_height: bottom_right = (bottom_right[0], image_height - 1)
            rectangle = [top_left, bottom_right]
            rectangles.append(rectangle)
            associated_keypoints.append(keypoint)

            facial_keypoints = [keypoint[17], keypoint[15], keypoint[0], keypoint[16], keypoint[18]]
            gaze_direction = gaze_2(facial_keypoints, keypoint)
            gazes.append(gaze_direction)

        else:
            continue

    return rectangles, gazes, associated_keypoints

def shot_type(keypoint):
    facial_keypoint_group = [keypoint[17][2], keypoint[15][2], keypoint[0][2], keypoint[16][2], keypoint[18][2]]
    body_keypoint_group = [keypoint[4][2], keypoint[3][2], keypoint[2][2], keypoint[1][2], keypoint[5][2],
                           keypoint[6][2], keypoint[7][2], keypoint[9][2], keypoint[8][2], keypoint[12][2],
                           keypoint[10][2], keypoint[13][2]]
    foot_keypoint_group = [keypoint[23][2], keypoint[22][2], keypoint[11][2], keypoint[24][2], keypoint[21][2],
                           keypoint[14][2], keypoint[19][2], keypoint[20][2]]

    if sum(facial_keypoint_group) > 0 and sum(body_keypoint_group) == 0 and sum(foot_keypoint_group) == 0:
        return 'close'
    if sum(facial_keypoint_group) > 0 and sum(body_keypoint_group) > 0 and sum(foot_keypoint_group) == 0:
        return 'medium'
    if sum(facial_keypoint_group) > 0 and sum(body_keypoint_group) > 0 and sum(foot_keypoint_group) > 0:
        return 'long'

    return 'unknown'

def extract_main_character(images_folder):
    main_character_dict = {"Filename": [], "Main Character Face Rectangles": [], "Shot Type": []};

    params = dict()
    params["model_folder"] = "openpose_models/"
    params["body"] = 1
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # user input parameters
    base_folder = images_folder + '/'

    file_list = [f for f in listdir(base_folder) if isfile(join(base_folder, f))]

    datum = op.Datum()
    file_counter = 1
    for filename in file_list:
        # Read image and face rectangle locations
        imageToProcess = cv2.imread(base_folder + filename)
        if imageToProcess is None:
            continue

        if imageToProcess.shape[1] > 3000:
            scale_percent = 50
        else:
            scale_percent = 100
        width = int(imageToProcess.shape[1] * scale_percent / 100)
        height = int(imageToProcess.shape[0] * scale_percent / 100)
        dim = (width, height)
        imageToProcess = cv2.resize(imageToProcess, dim, cv2.INTER_AREA)
        image_width = imageToProcess.shape[1]
        image_height = imageToProcess.shape[0]

        image_center_x = (image_width / 2)
        image_center_y = (image_height / 2)

        diagonal_over_2 = math.sqrt(image_width ** 2 + image_height ** 2) / 2

        # Create new datum
        datum.cvInputData = imageToProcess

        # Process and display image
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        keypoints = datum.poseKeypoints

        main_character_dict["Filename"].append(filename)

        main_character_face_rectangles = []

        shot = 'Empty'

        if keypoints is not None:
            facial_rectangles, gazes, associated_keypoints = face_rectangles(keypoints, image_width, image_height)
            if len(facial_rectangles) == 0:
                main_character_dict["Main Character Face Rectangles"].append(main_character_face_rectangles)
                main_character_dict["Shot Type"].append(shot)
                continue

            image_to_write = imageToProcess

            blur_values = []
            areas = []
            positionValues = []

            for rect in facial_rectangles:
                rect_center_x = (rect[0][0] + rect[1][0]) / 2
                rect_center_y = (rect[0][1] + rect[1][1]) / 2

                distance_to_center_x = abs(rect_center_x - image_center_x)
                distance_to_center_y = abs(rect_center_y - image_center_y)
                distance_to_center = math.sqrt(distance_to_center_x ** 2 + distance_to_center_y ** 2)

                positionValue = diagonal_over_2 - distance_to_center
                positionValues.append(positionValue)

                crop_img = image_to_write[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]

                if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                    blur = 0
                else:
                    blur = getBlurValue(crop_img)
                blur_values.append(blur)
                area = int(abs(rect[0][0] - rect[1][0]) * abs(rect[0][1] - rect[1][1]))
                areas.append(area)

            normGazeValues = []
            for gaze_direction in gazes:
                if gaze_direction == 'direct':
                    normGazeValues.append(1)
                if gaze_direction == 'right' or gaze_direction == 'left':
                    normGazeValues.append(1)
                if gaze_direction == 'undefined' or gaze_direction == 'away':
                    normGazeValues.append(0)

            blurImportance = 3
            areaImportance = 3.5
            positionImportance = 1.2

            blur_values = [a * b for a, b in zip(blur_values, normGazeValues)]
            areas = [a * b for a, b in zip(areas, normGazeValues)]
            positionValues = [a * b for a, b in zip(positionValues, normGazeValues)]

            if len(blur_values) == 1:
                blur_values[0] = 1
                areas[0] = 1
                positionValues[0] = 1

            if max(blur_values) == 0:
                normBlurs = [float('NaN')] * len(blur_values)
            else:
                normBlurs = [blurImportance * (blr / max(blur_values)) for blr in blur_values]

            for i in range(len(normBlurs)):
                if math.isnan(normBlurs[i]):
                    normBlurs[i] = 0
            if max(areas) == 0:
                normAreas = areas
            else:
                normAreas = [areaImportance * (i / max(areas)) for i in areas]

            if max(positionValues) == 0:
                normPositionValues = positionValues
            else:
                normPositionValues = [positionImportance * (i / max(positionValues)) for i in positionValues]

            normFocusValues = list(map(add, normBlurs, normAreas))
            normFocusValues = list(map(add, normFocusValues, normPositionValues))
            normFocusValues = [a * b for a, b in zip(normFocusValues, normGazeValues)]

            if len(normFocusValues) == 1:
                normFocusValues[0] = 1

            if max(normFocusValues) != 0:
                normFocusValues = [i / max(normFocusValues) for i in normFocusValues]

            predictedMainCharacters = []
            for normFocusValue in normFocusValues:
                if len(normFocusValues) == 2:
                    if normFocusValue > 0.86:
                        predictedMainCharacters.append(True)
                    else:
                        predictedMainCharacters.append(False)
                else:
                    if normFocusValue > 0.92:
                        predictedMainCharacters.append(True)
                    else:
                        predictedMainCharacters.append(False)

            gaze_index = 0

            rect_index = 0

            if True in predictedMainCharacters:
                keypoint_index_of_main_character = predictedMainCharacters.index(True)
                shot = shot_type(keypoints[keypoint_index_of_main_character])

                for rect in facial_rectangles:
                    if predictedMainCharacters[rect_index]:
                        main_character_face_rectangle = [100 / scale_percent *rect[0][0], 100 / scale_percent *rect[0][1], 100 / scale_percent *rect[1][0], 100 / scale_percent *rect[1][1]] # left top right bottom
                        main_character_face_rectangles.append(main_character_face_rectangle)
                    rect_index += 1
                    gaze_index += 1

        main_character_dict["Main Character Face Rectangles"].append(main_character_face_rectangles)
        main_character_dict["Shot Type"].append(shot)
        if file_counter%10==0: print(str(file_counter) + ' / 23502')
        file_counter += 1

    with open('data_pickle/' + images_folder_name + '/main_character.pickle', 'wb') as handle:
        pickle.dump(main_character_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rows = zip(main_character_dict['Filename'], main_character_dict['Main Character Face Rectangles'], main_character_dict['Shot Type'])
    with open('data_csv/' + images_folder_name + '/main_character.csv', "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(main_character_dict.keys())
        for row in rows:
            writer.writerow(row)

    return

def extract_gazes(images_folder):
    gaze_dict = {"Filename": [], "Face Rectangles": [], "Gaze": []};

    params = dict()
    params["model_folder"] = "openpose_models/"
    params["body"] = 1
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # user input parameters
    base_folder = images_folder + '/'

    file_list = [f for f in listdir(base_folder) if isfile(join(base_folder, f))]

    datum = op.Datum()
    file_counter = 1
    for filename in file_list:
        # Read image and face rectangle locations
        imageToProcess = cv2.imread(base_folder + filename)
        if imageToProcess is None:
            continue

        if imageToProcess.shape[1] > 3000:
            scale_percent = 50
        else:
            scale_percent = 100
        width = int(imageToProcess.shape[1] * scale_percent / 100)
        height = int(imageToProcess.shape[0] * scale_percent / 100)
        dim = (width, height)
        imageToProcess = cv2.resize(imageToProcess, dim, cv2.INTER_AREA)
        image_width = imageToProcess.shape[1]
        image_height = imageToProcess.shape[0]

        # Create new datum
        datum.cvInputData = imageToProcess

        # Process and display image
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        keypoints = datum.poseKeypoints

        gaze_dict["Filename"].append(filename)

        facial_rectangles = []

        gaze_directions = []

        if keypoints is not None:
            facial_rectangles, gazes, associated_keypoints = face_rectangles_2(keypoints, image_width, image_height)
            gaze_directions = gazes

        facial_rectangles_temp = []
        for rect in facial_rectangles:
            face_rectangle_rescaled = [100 / scale_percent *rect[0][0], 100 / scale_percent *rect[0][1], 100 / scale_percent *rect[1][0], 100 / scale_percent *rect[1][1]]
            facial_rectangles_temp.append(face_rectangle_rescaled)
        facial_rectangles = facial_rectangles_temp

        gaze_dict["Face Rectangles"].append(facial_rectangles)
        gaze_dict["Gaze"].append(gaze_directions)
        if file_counter%10==0: print(str(file_counter) + ' / 23502')
        file_counter += 1

    with open('data_pickle/' + images_folder_name + '/gazes.pickle', 'wb') as handle:
        pickle.dump(gaze_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rows = zip(gaze_dict['Filename'], gaze_dict['Face Rectangles'], gaze_dict['Gaze'])
    with open('data_csv/' + images_folder_name + '/gazes.csv', "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(gaze_dict.keys())
        for row in rows:
            writer.writerow(row)

    return

def extract_date_information(images_folder):
    date_dict = {"Filename": [], "Date": []};

    # user input parameters
    base_folder = images_folder + '/'

    file_list = [f for f in listdir(base_folder) if isfile(join(base_folder, f))]
    file_counter = 1
    for filename in file_list:
        path = base_folder + filename
        im = PIL.Image.open(path)
        exif = im.getexif()
        info = IPTCInfo(path)

        image_date = ''
        if bool(exif) is False and info['date created'] is None:
            image_date = ''
        if bool(exif) is True and 36867 in exif:
            date_created = exif.get(36867)
            split_date = date_created.split(':')
            year = split_date[0]
            month = split_date[1]
            day = split_date[2].split(' ')[0]
            image_date = day + ':' + month + ':' + year
        if ((bool(exif) is True and 36867 not in exif) or (bool(exif) is False)) and info['date created'] is not None:
            date_created = info['date created']
            date_created = date_created.decode('utf-8')
            year = date_created[0:4]
            month = date_created[4:6]
            day = date_created[6:8]
            image_date = day + ':' + month + ':' + year

        #creation_time = exif.get(36867)
        date_dict["Filename"].append(filename)
        date_dict["Date"].append(image_date)
        if file_counter%10==0: print(str(file_counter) + ' / 23502')
        file_counter += 1

    with open('data_pickle/' + images_folder_name + '/date.pickle', 'wb') as handle:
        pickle.dump(date_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rows = zip(date_dict['Filename'], date_dict['Date'])
    with open('data_csv/' + images_folder_name + '/date.csv', "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(date_dict.keys())
        for row in rows:
            writer.writerow(row)

    return

def extract_dimension_information(images_folder):
    #Dimension: width:height
    dimension_dict = {"Filename": [], "Dimensions": []};

    # user input parameters
    base_folder = images_folder + '/'

    file_list = [f for f in listdir(base_folder) if isfile(join(base_folder, f))]
    file_counter = 1
    for filename in file_list:
        path = base_folder + filename
        im = PIL.Image.open(path)

        dimensions = ''
        if im.size is not None:
            dimensions = str(im.size[0]) + ':' + str(im.size[1])

        #creation_time = exif.get(36867)
        dimension_dict["Filename"].append(filename)
        dimension_dict["Dimensions"].append(dimensions)
        if file_counter%10==0: print(str(file_counter) + ' / 23502')
        file_counter += 1

    with open('data_pickle/' + images_folder_name + '/dimensions.pickle', 'wb') as handle:
        pickle.dump(dimension_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rows = zip(dimension_dict['Filename'], dimension_dict['Dimensions'])
    with open('data_csv/' + images_folder_name + '/dimensions.csv', "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(dimension_dict.keys())
        for row in rows:
            writer.writerow(row)

    return

def get_model(cfg):
    base_model = getattr(applications, cfg.model.model_name)(
        include_top=False,
        input_shape=(cfg.model.img_size, cfg.model.img_size, 3),
        pooling="avg"
    )
    features = base_model.output
    pred_gender = Dense(units=2, activation="softmax", name="pred_gender")(features)
    pred_age = Dense(units=101, activation="softmax", name="pred_age")(features)
    model = Model(inputs=base_model.input, outputs=[pred_gender, pred_age])
    return model

def extract_age_gender(images_folder):
    age_gender_dict = {"Filename": [], "Age": [], "Gender": [], "Face Rectangles": []};

    weight_file = 'EfficientNetB3_224_weights.11-3.44.hdf5'
    margin = 0.1
    model_name, img_size = Path(weight_file).stem.split("_")[:2]
    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
    model = get_model(cfg)
    model.load_weights(weight_file)

    with open('data/gazes.pickle', 'rb') as handle:
        global gazes_dict
        gazes_dict = pickle.load(handle)

    base_folder = images_folder + '/'

    file_list = [f for f in listdir(base_folder) if isfile(join(base_folder, f))]
    file_counter=1
    for filename in file_list:
        # Read image and face rectangle locations
        imageToProcess = cv2.imread(base_folder + filename, 1)
        if imageToProcess is None:
            continue

        if imageToProcess is not None:
            h, w, _ = imageToProcess.shape
            r = 640 / max(w, h)
            imageToProcess = cv2.resize(imageToProcess, (int(w * r), int(h * r)))
            scale_percent = r*100

        face_rectangles = gazes_dict['Face Rectangles'][gazes_dict['Filename'].index(filename)]
        faces = np.empty((len(face_rectangles), img_size, img_size, 3))

        face_index = 0
        predicted_ages = []
        predicted_genders = []
        buggy_face_indexes = []
        for face_rect in face_rectangles:
            w = int(face_rect[3] - face_rect[1])
            h = int(face_rect[2] - face_rect[0])
            cropped_face_image = imageToProcess[int(face_rect[1]*scale_percent/100-h*margin):int(face_rect[3]*scale_percent/100+h*margin), int(face_rect[0]*scale_percent/100-w*margin):int(face_rect[2]*scale_percent/100+w*margin)]

            if cropped_face_image.size == 0:
                faces[face_index] = np.empty((img_size,img_size,3))
                buggy_face_indexes.append(face_index)
            else:
                faces[face_index] = cv2.resize(cropped_face_image, (img_size, img_size))
            face_index += 1

        if faces.shape[0] > 0:
            results = model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            predicted_genders_temp = []
            for predicted_gender in predicted_genders:
                if predicted_gender[0] > 0.5: pred_gend = 'F'
                else: pred_gend = 'M'
                predicted_genders_temp.append(pred_gend)
                predicted_genders = predicted_genders_temp

            if len(buggy_face_indexes) > 0 and len(predicted_genders) > 0 and len(predicted_ages) > 0:
                for buggy_index in buggy_face_indexes:
                    predicted_genders[buggy_index] = '-'
                    predicted_ages[buggy_index] = 0

        else:
            predicted_ages = []
            predicted_genders = []

        age_gender_dict['Filename'].append(filename)
        age_gender_dict['Age'].append(predicted_ages)
        age_gender_dict['Gender'].append(predicted_genders)
        age_gender_dict['Face Rectangles'].append(face_rectangles)
        if file_counter%10==0: print(str(file_counter) + ' / 23502')
        file_counter += 1

    with open('data_pickle/' + images_folder_name + '/age_gender.pickle', 'wb') as handle:
        pickle.dump(age_gender_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rows = zip(age_gender_dict['Filename'], age_gender_dict['Age'], age_gender_dict['Gender'], age_gender_dict['Face Rectangles'])
    with open('data_csv/' + images_folder_name + '/age_gender.csv', "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(age_gender_dict.keys())
        for row in rows:
            writer.writerow(row)

    return

class emotion_model():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']

        self.model = DAN(num_head=4, num_class=8)
        checkpoint = torch.load('affecnet8_epoch5_acc0.6209.pth',
                                map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.model.to(self.device)
        self.model.eval()

    def fit(self, np_img):
        img = PIL.Image.fromarray(np.uint8(np_img)).convert('RGB')
        img = self.data_transforms(img)
        img = img.view(1,3,224,224)
        img = img.to(self.device)

        with torch.set_grad_enabled(False):
            out, _, _ = self.model(img)
            _, pred = torch.max(out,1)
            index = int(pred)
            label = self.labels[index]

            return index, label

class DAN(nn.Module):
    def __init__(self, num_class=8,num_head=4):
        super(DAN, self).__init__()

        resnet = models.resnet18(True)

        checkpoint = torch.load('resnet18_msceleb.pth')
        resnet.load_state_dict(checkpoint['state_dict'],strict=True)

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_head
        for i in range(num_head):
            setattr(self,"cat_head%d" %i, CrossAttentionHead())
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)


    def forward(self, x):
        x = self.features(x)
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self,"cat_head%d" %i)(x))

        heads = torch.stack(heads).permute([1,0,2])
        if heads.size(1)>1:
            heads = F.log_softmax(heads,dim=1)

        out = self.fc(heads.sum(dim=1))
        out = self.bn(out)

        return out, x, heads

class CrossAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention()
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(sa)

        return ca

class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1,3),padding=(0,1)),
            nn.BatchNorm2d(512),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,1),padding=(1,0)),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU()


    def forward(self, x):
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1,keepdim=True)
        out = x*y

        return out

class ChannelAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid()
        )


    def forward(self, sa):
        sa = self.gap(sa)
        sa = sa.view(sa.size(0),-1)
        y = self.attention(sa)
        out = sa * y

        return out

def extract_emotion(images_folder):
    emotion_dict = {"Filename": [], "Emotion": [], "Face Rectangles": []};

    model = emotion_model()

    img_size = 224
    margin = 0.1

    with open('data/gazes.pickle', 'rb') as handle:
        global gazes_dict
        gazes_dict = pickle.load(handle)

    base_folder = images_folder + '/'

    file_list = [f for f in listdir(base_folder) if isfile(join(base_folder, f))]
    file_counter = 1
    for filename in file_list:
        # Read image and face rectangle locations
        imageToProcess = cv2.imread(base_folder + filename, 1)
        if imageToProcess is None:
            continue

        if imageToProcess is not None:
            h, w, _ = imageToProcess.shape
            r = 640 / max(w, h)
            imageToProcess = cv2.resize(imageToProcess, (int(w * r), int(h * r)))
            scale_percent = r*100

        face_rectangles = gazes_dict['Face Rectangles'][gazes_dict['Filename'].index(filename)]
        faces = np.empty((len(face_rectangles), img_size, img_size, 3))

        face_index = 0
        predicted_emotions = []
        buggy_face_indexes = []
        for face_rect in face_rectangles:
            w = int(face_rect[3] - face_rect[1])
            h = int(face_rect[2] - face_rect[0])
            cropped_face_image = imageToProcess[int(face_rect[1]*scale_percent/100-h*margin):int(face_rect[3]*scale_percent/100+h*margin), int(face_rect[0]*scale_percent/100-w*margin):int(face_rect[2]*scale_percent/100+w*margin)]

            if cropped_face_image.size == 0:
                faces[face_index] = np.empty((img_size,img_size,3))
                buggy_face_indexes.append(face_index)
            else:
                faces[face_index] = cv2.resize(cropped_face_image, (img_size, img_size))
            face_index += 1

        for face in faces:
            index, label = model.fit(face)
            predicted_emotions.append(label)

        if len(buggy_face_indexes) > 0 and len(predicted_emotions) > 0:
            for buggy_index in buggy_face_indexes:
                predicted_emotions[buggy_index] = '-'

        emotion_dict['Filename'].append(filename)
        emotion_dict['Emotion'].append(predicted_emotions)
        emotion_dict['Face Rectangles'].append(face_rectangles)
        if file_counter%10==0: print(str(file_counter) + ' / 23502')
        file_counter += 1

    with open('data_pickle/' + images_folder_name + '/emotions.pickle', 'wb') as handle:
        pickle.dump(emotion_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rows = zip(emotion_dict['Filename'], emotion_dict['Emotion'], emotion_dict['Face Rectangles'])
    with open('data_csv/' + images_folder_name + '/emotions.csv', "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(emotion_dict.keys())
        for row in rows:
            writer.writerow(row)

    return

def hue_to_color(hues, sats, values):
    color_counts = [0,0,0,0,0,0,0,0] #red, pink, orange, yellow, green, green-blue, blue, purple
    color_names = ['red', 'pink', 'orange', 'yellow', 'green', 'green-blue', 'blue', 'purple']
    hues = hues.flatten()
    sats = sats.flatten()
    values = values.flatten()
    satCutoff = 150
    valCutoff = 150
    color_counts[0] += np.count_nonzero((175 < hues) & (satCutoff < sats) & (valCutoff < values))#red
    color_counts[0] += np.count_nonzero((hues <= 6) & (satCutoff < sats) & (valCutoff < values))#red
    color_counts[1] = np.count_nonzero((155 < hues) & (hues <= 175) & (satCutoff < sats) & (valCutoff < values)) #pink
    color_counts[2] = np.count_nonzero((6 < hues) & (hues <= 23) & (satCutoff < sats) & (valCutoff < values)) #orange
    color_counts[3] = np.count_nonzero((23 < hues) & (hues <= 32) & (satCutoff < sats) & (valCutoff < values)) #yellow
    color_counts[4] = np.count_nonzero((32 < hues) & (hues <= 75) & (satCutoff < sats) & (valCutoff < values)) #green
    color_counts[5] = np.count_nonzero((75 < hues) & (hues <= 90) & (satCutoff < sats) & (valCutoff < values)) #green-blue
    color_counts[6] = np.count_nonzero((90 < hues) & (hues <= 123) & (satCutoff < sats) & (valCutoff < values)) #blue
    color_counts[7] = np.count_nonzero((123 < hues) & (hues <= 155) & (satCutoff < sats) & (valCutoff < values)) #purple

    if sum(color_counts) > 0.0 and max(color_counts)/len(hues) > 0.1:
        dominant_color_index = np.argmax(color_counts)
        dominant_color = color_names[dominant_color_index]

        color_counts[dominant_color_index] = 0
        dominant_color_2_index = np.argmax(color_counts)
        dominant_color_2 = color_names[dominant_color_2_index]
    else:
        dominant_color = '-'
        dominant_color_2 = '-'

    return dominant_color, dominant_color_2

def extract_color_information(images_folder):
    color_dict = {"Filename": [], "Shape": [], "TonalityDark": [], "DominantColor1": [], "DominantColor2": [], 'Greyscale': []};

    base_folder = images_folder + '/'

    file_list = [f for f in listdir(base_folder) if isfile(join(base_folder, f))]
    file_counter = 1
    for filename in file_list:
        imageToProcess = cv2.imread(base_folder + filename, 1)
        isGreyscale = False
        if len(imageToProcess.shape) < 3: isGreyscale = True
        if len(imageToProcess.shape) == 3:
            if imageToProcess.shape[2]  == 1: isGreyscale = True
            else:
                b,g,r = imageToProcess[:,:,0], imageToProcess[:,:,1], imageToProcess[:,:,2]
                if (b==g).all() and (b==r).all(): isGreyscale = True

        if not isGreyscale:
            hsv = cv2.cvtColor(imageToProcess, cv2.COLOR_BGR2HSV)
            h = hsv[:,:,0]
            s = hsv[:,:,1]
            v = hsv[:,:,2]
            dominantColor, dominantSecondColor = hue_to_color(h, s, v)
        else:
            dominantColor = 'grey'
            dominantSecondColor = 'grey'
        lowContrast = is_low_contrast(imageToProcess, fraction_threshold=0.3)
        width = imageToProcess.shape[1]
        height = imageToProcess.shape[0]
        widthToHeightRatio = width/height
        if 0.85 <= widthToHeightRatio and widthToHeightRatio <= 1.15:
            imageShape = 'square'
        if widthToHeightRatio < 0.85:
            imageShape = 'vertical'
        if widthToHeightRatio > 1.15:
            imageShape = 'horizontal'

        color_dict['Filename'].append(filename)
        color_dict['Shape'].append(imageShape)
        color_dict['TonalityDark'].append(lowContrast)
        color_dict['DominantColor1'].append(dominantColor)
        color_dict['DominantColor2'].append(dominantSecondColor)
        color_dict['Greyscale'].append(isGreyscale)
        if file_counter%10==0: print(str(file_counter) + ' / 23502')
        file_counter += 1

    with open('data_pickle/' + images_folder_name + '/colors.pickle', 'wb') as handle:
        pickle.dump(color_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rows = zip(color_dict['Filename'], color_dict['Shape'], color_dict['TonalityDark'], color_dict['DominantColor1'], color_dict['DominantColor2'], color_dict['Greyscale'])
    with open('data_csv/' + images_folder_name + '/colors.csv', "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(color_dict.keys())
        for row in rows:
            writer.writerow(row)

    return

def main(path, method):
    images_folder = path
    if not os.path.isdir('data_pickle'):
        os.mkdir('data_pickle')
    if not os.path.isdir('data_csv'):
        os.mkdir('data_csv')

    global images_folder_name
    images_folder_name = images_folder.split('/')[-1]

    if not os.path.isdir('data_pickle/' + images_folder_name):
        os.mkdir('data_pickle/' + images_folder_name)
    if not os.path.isdir('data_csv/' + images_folder_name):
        os.mkdir('data_csv/' + images_folder_name)

    if method == "all" or method == "face":
        extract_face_information(images_folder)
    if method == "all" or method == "similarity":
        extract_image_vectors(images_folder)
    if method == "all" or method == "scenes":
        extract_scene_information(images_folder)
    if method == "all" or method == "objects":
        extract_object_information(images_folder)
    if method == "all" or method == "captions":
        extract_image_captions(images_folder)
    if method == "all" or method == "main_characters":
        extract_main_character(images_folder)
    if method == "all" or method == "gazes":
        extract_gazes(images_folder)
    if method == "all" or method == "date":
        extract_date_information(images_folder)
    if method == "all" or method == "dimensions":
        extract_dimension_information(images_folder)
    if method == "all" or method == "age_gender":
        extract_age_gender(images_folder)
    if method == "all" or method == "emotions":
        extract_emotion(images_folder)
    if method == "all" or method == "colors":
        extract_color_information(images_folder)

    return

if __name__ == "__main__":
    main()