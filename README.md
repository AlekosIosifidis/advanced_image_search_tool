# advanced_image_search_tool
Machine learning based advanced image search tool
## Contents
1. [Getting Started](#getting-started)
2. [Easy Install](#easy-install)
3. [Extracting Data](#extracting-data)
4. [Using the Search Tool](#using-search-tool)
5. [Working with Multiple Image Directories](#working-with-multiple-image-directories)

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
Nvidia CUDA (recommended version 10.1) should also be installed.

The following models should also be installed from their respective repositories:

- affecnet8_epoch5_acc0.6209.pth, resnet18_msceleb.pth (https://github.com/yaoing/DAN)
- BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar (https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
- coco_weights.pt, conceptual_weights.pt (https://github.com/rmokady/CLIP_prefix_caption)
- dlib_face_recognition_resnet_model_v1.dat, shape_predictor_5_face_landmarks.dat (https://github.com/ageitgey/face_recognition_models/tree/master/face_recognition_models/models)
- EfficientNetB3_224_weights.11-3.44.hdf5 (https://github.com/yu4u/age-gender-estimation)
- resnet18_places365.pth.tar (https://github.com/CSAILVision/places365)

## Easy Install
In case you don't wish to download the required models individually or if some models are no longer available, you can simply download [SearchTool1.zip](SearchTool1.zip), [SearchTool2.zip](SearchTool2.zip) and [SearchTool3.zip](SearchTool3.zip). These zip files contains all of the code, dll files and models necessary to run the program. Then, you need to extract the contents of all 3 zip files under the same working directory. You will still need to install the required libraries mentioned in section [Getting Started](#getting-started) but you can skip the installation of the models as they will be included within the zip files.

## Extracting Data
Before running the search tool, you need to create a folder under the working directory and put all of your image files under that folder. This folder will act as a search corpus from where you can search and find images with the tool. Then, you need to run the script "extract_information_UI.py". The UI will look like this:

<p float="center">
  <img src="https://github.com/mertseker-dev/advanced_image_search_tool/blob/main/extract_information.JPG" width="100" />
</p>

From here, you first need to click on "Choose Folder" and choose the folder that contains your images. Note that this folder should be put under the working directory.
Once the folder is chosen, the folder's name will appear with green text under the "Choose Folder" button.

You can then choose to extract individual information by clicking one of the 12 buttons (from "Face" until "Main Characters") or you can click on "Extract All" to extract all of them at once.

The extracted information will be stored in both .csv and .pickle format. You can find them under data_csv/{your_image_folder} and data_pickle/{your_image_folder}.

## Using The Search Tool

The search tool's UI looks like this:

<p float="center">
  <img src="https://github.com/mertseker-dev/advanced_image_search_tool/blob/main/searchTool1.JPG" width="800" />
</p>

First, you need to click on the "Choose Folder" button on the top left and choose the folder under the working directory that contains your images. After you select it, the folder's name should appear with green text next to the "Choose Folder" button.

All of the filters work in combination with each other (AND logic). If you were to use multiple filters, images that fit to all of the filters you have chosen will show up as results.

Here's the explanation of each feature:

- Content based search: Here, you can directly type in a descriptive sentence of the images you wish to find. For example, you can type "soldiers standing in front of a lake". Images that fit to your description will show up as results.
- Search similar images: You can click on the "Search similar images" button and select an example/input image. After the image is processed, a green text that says "File chosen" will appear on the right side of the button. Images that look similar to the input image will show up as results.
- Search by face: You can click on the "Search by face" button and select an input image. After the image is processed, a green text that says "File chosen" will appear on the right side of the button. The file should contain at least 1 visible face. The faces will automatically be detected and shown with blue rectangles on a pop-up window. You can choose one of the faces by clicking on any point within that face's rectangle (face rectangle will turn from blue to green when you hover your cursor in them). Then, images matching with the selected face will show up as results.
- Search by shot type: You can choose one or more of the following shot type options: "Close", "Medium" and "Long". Images with shot types of your selections will show up as results.
- Search by environment type: You can choose either "Indoor" or "Outdoor". Images with your selected environment type will show up as results.
- Search by number of people: You can type in a positive integer number in the text box. Images with number of people that match exactly with the number you have entered will show up as results. For example, if you typed in 3, images containing exactly 3 people will show up.
- Search by date: You can choose a date range here. You may also choose only a beginning or an ending date. Images that were taken within the date range you have selected will show up as results.
- Search by size: Here, you can enter pixel width and height and then specify one of the options: "Smaller than", "Exactly" or "Larger than". Smaller and larger than options work on both dimension. For example, if you were to enter 1000 and 1000 as width and height respectively and select smaller than, images with less than 1000 width and less than 1000 height will show up (an image with size 900x900 would show up but 900x1100 won't).
- Search by object: Here, you can either search for objects by typing their names in the text box or you can scroll through the menu to find them. You may choose multiple objects. Images that contain at least 1 of all your chosen objects will show up as results. The "CLEAR" button will clear all of your selections with this feature.
- Search by scene: Here, you can either search for scenes by typing their names in the text box or you can scroll through the menu to find them. You can only choose one scene type. Images that match with your chosen scene type will show up as results. The "CLEAR" button will clear all of your selections with this feature.
- Search by gaze: Here you can choose one of the options: "Left", "Direct", "Right", "Away". Images where the subjects are looking to the direction you have chosen will show up as results. Note that these directions are from the perspective of you, the image viewer and not the subjects themselves. This means "Left" refers to the left side of the image.
- Search by age: Here you can choose one or more of the following four age groups: "Children", "Youth", "Adult" and "Senior". Images where the subjects are matching with the age groups you have selected will show up as results.
- Search by gender: Here you can choose one of the following genders: "Male", "Female". Images where the subjects are matching with the genders you have selected will show up as results.
- Search by age: Here you can choose one or more of the following eight emotion groups: "Neutral", "Happy", "Sad", "Fear", "Surprise", "Disgust", "Anger" and "Contempt". Images where the emotions of the subjects are matching with the emotion groups you have selected will show up as results.
- Main character, Anyone, Everyone: These 3 options apply to the following four filters: "Search by gaze", "Search by age", "Search by gender" and "Search by emotion". At least one of these 3 options must be seleted for these four filters to work. These selections specify to which subjects these four filters should apply to. Main character refers to the most important person in the image, anyone refers to at least 1 person and everyone refers to all of the subjects in the image. For example if you were to select "Left", "Adult", "Female" and "Happy" in these four filters and select "Main Character", images where the main character is an adult female who is happy and looking to the left will show up. If you select "Anyone", images where at least one person who fit the filters will show up, the person does not need to be a main character in this case. If you select "Everyone", images where all of the subjects fit the filters will show up.
- Search by shape: Here you can select one of the three following options: "Square", "Horizontal" and "Vertical". These shapes refers to pixel-wise dimensions of the images.
- Search by color: Here you can select one of the two following options: "Color" and "Greyscale". These refer to the color scheme of the images.
- Search by contrast: Here you can select one of the two following options: "Low" and "Normal". You can use this options to find low or normal contrast images.
- Dominant colors: Here you can select on of the eight color groups. Images where the dominant color is your selected color will show up as results.
- CLEAR: The "CLEAR" button on the bottom will completely clear and reset the filter selections you have made (except for the images folder, which will not be reset).
- SEARCH: The "SEARCH" button will apply all of the filters you have selected and search for images that fit to all of your filters within your image folder. The results will appear under "search results/{your_image_folder}" and this folder will automatically show up as a maximized window after the search operation is complete. **IMPORTANT** Note that the results under this directory will be reset after each operation you make, which means the results are only valid for the latest search operation you have made. If you wish to save the results of your search, you should take a backup of them before you perform a new search.

## Working with Multiple Image Directories
If you wish to work with multiple image directories (multiple image search corpuses), you need to put all of your image folders as separate folders under the working directory. For example, if you were using 2 search corpuses, they should appear as {working_directory}/{your_image_folder_1} and {working_directory}/{your_image_folder_2}. You should select your image folder both on "extract_information_UI.py" and "search.py" as explained in the sections above. The extracted information will appear under "data_csv/{your_image_folder}", "data_pickle/{your_image_folder}" and the search results will appear under "search results/{your_image_folder}".






