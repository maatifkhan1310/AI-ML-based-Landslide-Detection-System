# AI-Based Landslide Prediction and Monitoring System  

## Overview  
This repository contains the source code, datasets, and documentation for the **AI-Based Landslide Prediction and Monitoring System**. The system predicts landslide-prone areas based on geographical images and terrain data using a machine learning model, specifically a Convolutional Neural Network (CNN).  

## Features  
- Automated image analysis to detect potential landslide zones.  
- Utilizes a CNN for high-accuracy predictions.  
- Scalable architecture for both offline and real-time analysis.  
- Future scope includes integration with IoT sensors for environmental data.  

## Project Structure  
- **datasets/**: Contains training and testing data.  
- **models/**: Includes ML model scripts and pre-trained weights.  
- **notebooks/**: Jupyter Notebooks for data preprocessing, training, and evaluation.  
- **utils/**: Helper scripts for data loading, image processing, and visualization.  

## Datasets  
The datasets used for this project consist of image data from landslide-prone and non-landslide-prone areas.  

- **Primary Dataset:** [Kaggle Landslide Dataset](https://www.kaggle.com/competitions/landslide-detection-data)  
- **Supplementary Data:** Satellite and terrain images from [USGS Earth Explorer](https://earthexplorer.usgs.gov/).  

## Technologies Used  
- Python  
- TensorFlow/Keras  
- OpenCV  
- Scikit-learn  
- Matplotlib/Seaborn  

## Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/AI-Based-Landslide-Prediction-System.git

## Usage
Run the preprocessing script:
        python utils/data_loader.py
        
## Train the model:
        python models/training_script.py
        
## Evaluate the model:
        python models/evaluation.py
        
## Results
Achieved 85% accuracy on the test set using the CNN model.
