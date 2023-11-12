# Parasitic_Egg_Multiclass_Object_Detection_and_Counting_in_Microscopic_Images

## Background & Problem Statement
Parasitic infections have been recognized as one of the most significant causes of illnesses by WHO. Diarrheal disease, which is often caused by parasitic infection is the TOP 5 leading cause of children death in low-income countries.

Diagnosis of intestinal parasites is usually based on direct microscopic examination, which require trained/skilled personnel that often lacking in remote region.

THAT IS WHY we need a new tool for diagnosis that is easier to perform even by any person without medical/laboratory capacity, and have a very good sensitivity and specificity. 
![image](https://github.com/zakky211/Parasitic_Egg_Multiclass_Object_Detection_and_Counting_in_Microscopic_Images/assets/62234134/247b49e1-0735-43de-8c3b-4bd9e745ab96)

## Objective
To develop robust algorithm to detect eggs of parasitic worms in a variety of microscopic images that is easier to perform even by any person without medical/laboratory capacity, and have a very good sensitivity and specificity. 

## Data Collection & Preparation
Dataset taken from : 
[Parasitic Egg Detection and Classification in Microscopic Images(ieee-dataport.org)](https://ieee-dataport.org/competitions/parasitic-egg-detection-and-classification-microscopic-images#files)

Dataset contain 11 parasitic egg types, each has 1,000 images :
category 0: Ascaris lumbricoides
category 1: Capillaria philippinensis
category 2: Enterobius vermicularis
category 3: Fasciolopsis buski
category 4: Hookworm egg
category 5: Hymenolepis diminuta
category 6: Hymenolepis nana
category 7: Opisthorchis viverrine
category 8: Paragonimus spp
category 9: Taenia spp. egg
category 10: Trichuris trichiura
Annotation Format : COCO JSON
![image](https://github.com/zakky211/Parasitic_Egg_Multiclass_Object_Detection_and_Counting_in_Microscopic_Images/assets/62234134/de757185-95f8-4539-a6bb-25d14be2ecee)

![image](https://github.com/zakky211/Parasitic_Egg_Multiclass_Object_Detection_and_Counting_in_Microscopic_Images/assets/62234134/65a508ba-6ac0-4596-9384-f4f51c6eca3a)

## Data Preprocessing
Dataset comes with default preprocessed images :

Apply Gaussian blur with a standard deviation between 0.0 - 3.0
Crop each side by a random value from 0 to 30%
Apply motion blur (10% of total images) with a random kernel size of 3- 35 pixels and orientation of 0-360 degree
Apply Gaussian noise with a standard deviation between 0.0 - 25.5 (to each colour channel separately)
Apply Poisson noise with a lambda (the expected rate of occurrences) between 0.0 - 5.0
Adjust image saturation by adding a value in a range of -25 - +25 to the S channel of the HSV colour space
Adjust contrast using gamma correction with a gamma value between 0.5 - 2.0

## Modelling

Model 	  	: YOLOv5m
Split 	  	: Train 8800 (800 per category), Val 2200 (200 per category), Test  2200
Batch 		  : 12
Epochs   	  : 40
Workers   	: 32
Optimizer 	: SGD

## Result

![image](https://github.com/zakky211/Parasitic_Egg_Multiclass_Object_Detection_and_Counting_in_Microscopic_Images/assets/62234134/2219e7a4-0663-42af-a60a-e2ec7807f5a5)

Confusion Matrix
![confusion_matrix](https://github.com/zakky211/Parasitic_Egg_Multiclass_Object_Detection_and_Counting_in_Microscopic_Images/assets/62234134/ea7ca45b-bfb6-4a1c-9680-6da66fdfeb4f)

Annotation
![val_batch0_labels](https://github.com/zakky211/Parasitic_Egg_Multiclass_Object_Detection_and_Counting_in_Microscopic_Images/assets/62234134/b9f81077-2911-4bc5-967b-9ddf98d4482b)

Prediction
![val_batch0_pred](https://github.com/zakky211/Parasitic_Egg_Multiclass_Object_Detection_and_Counting_in_Microscopic_Images/assets/62234134/8f850d2f-d841-4a94-9c36-081f29cddcfa)

# Deploy to Web Streamlit
![image](https://github.com/zakky211/Parasitic_Egg_Multiclass_Object_Detection_and_Counting_in_Microscopic_Images/assets/62234134/057818a0-cccc-4e22-95c6-012622be2106)
