# 3D Print Quality Checker

This project was developed as my capstone project, designed to offer an easy-to-use tool for hobbyists, makers, and small businesses to quickly identify common defects in their 3D printed objects. The tool is accessible directly through a web browser, making defect detection simple and convenient.

## Project Goal

The primary goal of this tool is to allow users to upload images of their completed 3D prints and receive immediate analysis highlighting any defects such as:

- Uneven layers  
- Stringing  
- Alignment issues  
- Other common 3D printing errors  

The aim is to help users improve print quality without requiring specialized knowledge in machine learning or computer vision.

## Installation
1. Download Docker or Docker-Desktop
2. Clone repository to your pc
3. Download the models from [releases](https://github.com/avkoll/3DPQC/releases/tag/model_binariesV1.0) and unzip the files into the /models directory of the project.
4. Start docker
5. Navigate to project root in commandline and type: docker-compose build (This may take a while.)
6. Once that is done run: docker-compose up
7. Navigate to http://127.0.0.1:5000 in your browser

## How It Works

1. Users access the tool via a simple web interface.
2. They upload images of their finished prints.
3. Computer vision (CV) models, trained to detect specific types of defects, analyze the images.
4. The tool provides feedback directly in the browser, clearly indicating any detected flaws.

## Technology Stack

- **Python (Flask)** – Backend server to handle image uploads and manage model inference.  
- **OpenCV** – Image preprocessing and manipulation.  
- **TensorFlow** – Training and evaluation of machine learning models.  
- **Docker & Docker Compose** – Each component of the project (Flask backend and individual CV models) runs in its own Docker container, allowing for easy deployment and scalability.

## Model Training & Performance

- **Dataset**:  
  The model was trained using a dataset retrieved from [this Kaggle repository](https://www.kaggle.com/datasets/tangyiqi/3d-print-error-images-after-data-enhancement).

- **Initial Model (SVM with RBF kernel)**:  
  - Parameters: `C = 1.0`, `gamma = 0.001`  
  - Accuracy: ~23%

- **Current Model (TensorFlow-based)**:  
  - Best Parameters:  
    - Kernel: `rbf`  
    - Gamma: `0.00078125`  
    - Degree: `3`  
    - Coef0: `1.0`  
    - Class Weight: `balanced`  
    - C: `100`  
  - Validation Accuracy: **64.27%**

The trained model is available in the release tab of this repository.

## Limitations & Future Work

The current model is designed to detect **one flaw at a time**.  
My original concept involved **separate models for different types of defects**, which would allow the system to identify **multiple flaws within a single image**. This approach remains a potential area for future improvement, as it could increase both the granularity and accuracy of defect detection.
