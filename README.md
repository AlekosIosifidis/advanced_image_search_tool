# advanced_image_search_tool
Machine learning based advanced image search tool
## Contents
1. [Getting Started](#getting-started)
2. [Easy Install](#easy-install)
3. [Extracting Data](#extracting-data)
4. [Using the Search Tool](#using-search-tool)
5. [Viewing Results](#viewing-results)

## Getting Started
The code requires the following libraries to be installed:

-  python 3.8
-  tensorflow 2.3.1
-  opencv 4.4.0.44
-  numpy 1.18.5
-  tk 0.10.0
-  imgsim 0.1.1
-  dlib 19.23.1
-  sklearn 0.0
-  google-trans-new 1.1.9
-  torch 1.9.0+cu102
-  sentence-transformers 2.0.0
-  IPTCInfo3 2.1.4
-  omegaconf 2.1.1
-  imageio 2.9.0
-  clip 1.0
-  typing 3.7.4.3
-  transformers 4.9.1

The code requires OpenPose to be installed. Refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose for installation instructions.

The following models should also be installed from their respective repositories:

- affecnet8_epoch5_acc0.6209.pth, resnet18_msceleb.pth (https://github.com/yaoing/DAN)
- BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar (https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
- coco_weights.pt, conceptual_weights.pt (https://github.com/rmokady/CLIP_prefix_caption)
- dlib_face_recognition_resnet_model_v1.dat, shape_predictor_5_face_landmarks.dat (https://github.com/ageitgey/face_recognition_models/tree/master/face_recognition_models/models)
- EfficientNetB3_224_weights.11-3.44.hdf5 (https://github.com/yu4u/age-gender-estimation)
- resnet18_places365.pth.tar (https://github.com/CSAILVision/places365)

## Easy Install
In case you don't wish to download the required models individually or if some models are no longer available, you can 


