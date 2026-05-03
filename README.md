# Emoji Image Classifier – DS3273 Final Project

This repository contains my final project for DS3273 (Jan–2026).  
It implements a CNN-based classifier for emoji images and follows the exact
project structure required in the project_submission_format.pdf.

---

## Installation

1. Clone this repository:
   git clone <your-public-repo-link>

2. Install dependencies:
   pip install torch torchvision

No additional packages are required.

---

## How to Train

Run:
python train.py

This trains the CNN_Emoji model and saves the weights to:
checkpoints/final_weights.pth

---

## How to Predict

Use the provided prediction function:

from predict import the_predictor
the_predictor(["data/img01.png"])

This returns a list of predicted class indices.

---

## Project Structure

- model.py — defines CNN_Emoji  
- dataset.py — loads images from data/  
- train.py — training loop  
- predict.py — batch inference function  
- interface.py — exposes required names for grading  
- config.py — hyperparameters  
- data/ — contains 10 raw images  
- checkpoints/ — contains final_weights.pth  

---

## Notes

- The repository is publicly accessible as required.
- README includes installation and execution instructions.
- All files follow the naming and structure rules from the PDF.
