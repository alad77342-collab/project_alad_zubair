# Emoji Image Classifier – Final Project

This project implements a CNN-based classifier for emoji images, following the required structure from the project_submission_format.pdf.

## Project Structure
- model.py — defines the CNN_Emoji model  
- dataset.py — loads images from the data/ directory  
- train.py — trains the model and saves checkpoints/final_weights.pth  
- predict.py — contains the_predictor() for batch inference  
- interface.py — exposes required names for grading  
- config.py — stores hyperparameters  

## How to Train
Run:
python train.py

## How to Predict
from predict import the_predictor  
the_predictor(["data/1.jpg"])

## Notes
- All images in data/ are raw, unmodified.  
- The project follows the exact directory and naming rules required for grading.
