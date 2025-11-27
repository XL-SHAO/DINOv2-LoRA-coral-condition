# Coral image dataset for multi-temporal, multi-spatial coral conditions monitoring in the Indo-Pacific
Dataset, labels, and codes for coral condition classification paper: Multi-label classification for multi-temporal, multi-spatial coral reef condition monitoring using vision foundation model with adapter learning
## Project
### Project title
The Koh Tao Coral Condition Survey Project was initiated by Jun Sasaki’s laboratory, University of Tokyo, and implemented in collaboration with the New Heaven Reef Conservation Program, Koh Tao, Thailand.
### Project description
Given the Koh Tao Island case context, this study is the first to explore the efficient adaptation of vision foundation models for multi-label classification of coral reef conditions under multi-temporal and multi-spatial settings. It aims to revolutionize the current monitoring program by integrating underwater images, citizen science conservation knowledge, and vision foundation models. Coral images were collected by divers holding an Olympus TG-6 with waterproof housing. Geographical parameters were collected as metadata and saved as tabular data of the supplementary materials.
## Dataset
### Dataset description
The project covered most of the coral habitats surrounding Koh Tao, selected based on the weather conditions and characteristics of the habitats. The dataset consists of (1) the original, uncropped 3000*4000 pixels images; (2) cropped, 512x512 pixels image patches for the training and test of the model; (3) classification labels annotated by human experts; and (4) metadata of each field survey.
![Study area](assets/study_area.png)
### Dataset organization
There are 42,105 image patches generated from 40 times of field surveys. At each site, the sampling strategy is as follows:
![sampling](https://github.com/XL-SHAO/CoralConditionDataset/assets/117028875/9f417cd5-7aea-4cb7-b18b-d28706c904e9)
All the image patches are in JPG format. The images are organized based on the survey they were collected from. For example, the folder ‘20230804_CBK’ is named after the survey date_abbreviation of the site. The full name of each site can be found in the ‘surveys_metadata’ in the tabular_data folder.
### Image naming conventions
The name of each image consists of five parts separated by underscores: the three-letter site code of the sampling site (e.g., ALK=Aow Leuk, CBK=Chalok Baan Kao Bay), the unique four-digit survey number, the two-digit transect number, the date of the survey formatted as YYYYMMDD, the four-digit image number, and the number of the image patch. Examples of the images are as follows:
CBK_0001_11_20230805_0001_12, TTB_0002_00_20230815_0002_02
### Dataset content
**1.Coral images:** All image patches are in 512x512 pixels. The file [coral_images_dry.zip](https://drive.google.com/drive/folders/1yjvVGSXuFRcO3b0SehyeAHzMtCHhI6S1?usp=drive_link) contains underwater coral images taken during April 2023 and August to September 2023, representing the dry season. The file [coral_images_wet.zip](https://drive.google.com/file/d/1cbmTZ9-y7B51PPFyStSZhJNgeqD7WTwA/view?usp=sharing) contains coral images taken from January to February 2024, representing the wet season.

**2.Metadata:** (surveys_metadata.csv) this file provides additional information about the images, and it is organized by the survey id, which refers to every single time of the field observation. The parameters contained in this table are as follows:
* surveyid: the unique four-digit code representing each observation. For instance, the first field observation was conducted on August 4th, 2023, at Chalok Baan Kao Bay; all the images from this observation will be named with survey id 0001.
* transectid: the two-digit code representing the transect lines. The first digit identifies if the transect is permanent (1-permanent, 0-temporal), and the second digit reflects if the transect is the 1-shallow line (2-4m depth) or 2-deep line (9-10m depth). Additionally, 00 stands for images collected randomly without setting up transects.
* survey_date: in YYYYMMDD form, reflecting the date the survey is conducted.
* site: the three-letter site identification code representing which diving site those data were collected from.
* folder_name: name of the folder where the images were saved.
* lat_start, lng_start: the latitude and longitude of the starting point of this survey.
* camera: camera used to capture the images.
* depth: average depth of this survey.
* temp: average temperature of this survey.

**3.Labelset of the annotation:** (labelsets.csv) this table contains the set of possible labels that can be assigned to the objects within the images. It includes four labels reflecting the coral conditions and four for stressors, which are 1 = Healthy coral, 2 = Compromised coral, 3 = Dead coral, 4 = Rubble, 5 = Competition, 6 = Disease, 7 = Predation, and 8 = Physical issues

**4.Annotations:** (annotations.csv) annotation file contains the image patch id and its corresponding classification label, which is used for model training and image classification. This annotation was labeled by human experts in marine ecology and coral conservation.

## Methodology
An DINOv2 model fine-tuned with LoRA adapter was proposed to automatically classify coral images with multiple labels, following the classification standard used in coral reef conservation programs. The adapter learning with LoRA significantly reduced the number of trainable parameters of the proposed model, resulting in high computational efficiency. Codes are available in this repository, and the overall framework of this proposed approach is illustrated as follows:
![framework](assets/proposed_method.png)
# Citation
If this dataset or codes contributes to your research, please consider citing our paper:
```LaTeX
@article{SHAO2026DINOv2LoRA,
title = {Multi-label classification for multi-temporal, multi-spatial coral reef condition monitoring using vision foundation model with adapter learning},
journal = {Marine Pollution Bulletin},
volume = {223},
pages = {119054},
year = {2026},
issn = {0025-326X},
doi = {https://doi.org/10.1016/j.marpolbul.2025.119054},
url = {https://www.sciencedirect.com/science/article/pii/S0025326X25015309},
author = {Xinlei Shao and Hongruixuan Chen and Fan Zhao and Kirsty Magson and Jundong Chen and Peiran Li and Jiaqi Wang and Jun Sasaki}
}
```
# Q & A
For any questions, please [contact us.](mailto:yuishaoxinlei@gmail.com)
