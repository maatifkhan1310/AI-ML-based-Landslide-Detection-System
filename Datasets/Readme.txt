# Training Data  

## Overview  
This folder contains the training dataset used for the AI-Based Landslide Prediction System. The data is categorized into two classes:  
1. **Landslide:** Images depicting areas prone to or affected by landslides.  
2. **Non-Landslide:** Images of stable terrains with no visible landslide activity.  

## Folder Structure  
- **landslide/**: Contains images labeled as landslide-prone.  
- **non_landslide/**: Contains images labeled as non-landslide-prone.  

## Image Specifications  
- Format: JPEG/PNG  
- Dimensions: 128x128 (preprocessed for the model)  
- Color mode: RGB  

## Usage  
Ensure that the dataset is split into training and testing sets before model training. Use an 80/20 split for optimal results.  

## Preprocessing  
Images were resized to 128x128 pixels and normalized to improve training efficiency.  
