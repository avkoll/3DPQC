# 3D Print Quality Checker

This project is developed as my capstone project aimed at providing an easy-to-use tool for hobbyists, makers, and small businesses to quickly identify common defects in their 3D printed objects. The tool is designed to be simple and accessible directly through a web browser.

## Project Goal
The primary goal of this tool is to enable users to upload images of their completed 3D prints and receive immediate analysis highlighting any defects such as uneven layers, stringing, alignment issues, and other common printing errors.

## How it Works
- Users access the tool via a simple web interface.
- Images of finished prints are uploaded for analysis.
- Computer vision (CV) models, each specialized in detecting specific types of defects, analyze the images.
- Feedback is provided directly in the browser, clearly indicating any detected flaws.

## Technology Stack
- Python (Flask)
- OpenCV
- Docker and Docker-Compose

Each component of the project, including the Flask backend and individual CV models, runs in its own Docker container, managed with Docker Compose to ensure easy deployment and scalability.


