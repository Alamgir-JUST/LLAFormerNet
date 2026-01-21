# LLAFormerNet: A Lightweight Hybrid Deep Learning Architecture for Anomaly Detection in Containerized Environments

## Overview

LLAFormerNet is a novel deep learning-based approach designed for anomaly detection and misuse detection in containerized environments, specifically targeting cloud-native systems. This repository contains the full implementation of the proposed model, which combines 1D Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTM) networks, and Temporal Convolutional Networks (TCNs), with an integrated attention mechanism. The model efficiently captures both local and long-range temporal dependencies in container network traffic to provide real-time, accurate anomaly detection.

## Key Features

-Hybrid Architecture: Combines CNN, LSTM, and TCN to detect complex patterns in containerized network traffic.
-Efficient for Real-time Detection: Lightweight design, optimized for minimal computational overhead while maintaining high accuracy.
-Interpretability: Incorporates an attention mechanism, enabling users to understand which features are influencing the detection decision.
-Extensive Evaluation: Evaluated on a real-world dataset (Misuse Detection in Containers Dataset, AINA 2024) with state-of-the-art results.

## Model Architecture
<p align="center"> <img src="https://github.com/Alamgir-JUST/LLAFormerNet/blob/1ebb8ce6497aba24635341a1c1dae3ec323d9529/Development%20Pipeline.png"/> </p>
<p align="center"> <img src="https://github.com/Alamgir-JUST/LLAFormerNet/blob/1ebb8ce6497aba24635341a1c1dae3ec323d9529/llatformer_model.h5%20(1).png"/> </p>

## Installation

To run this code, clone the repository and install the required dependencies using the following steps:

git clone https://github.com/Alamgir-JUST/LLAFormerNet.git
cd LLAFormerNet
pip install -r requirements.txt

##Requirements

-Python 3.x
-TensorFlow 2.x
-NumPy
-Pandas
-Scikit-learn

## Results

The model has achieved state-of-the-art results in binary anomaly detection with the following metrics:
-Accuracy: 99.89%
-Precision (weighted): 99.89%
-Recall (weighted): 99.89%
-F1 Score (weighted): 99.89%

Additionally, confidence intervals for these metrics are included to ensure robustness.

## Confusion Matrix
<p align="center"> <img src="https://github.com/Alamgir-JUST/LLAFormerNet/blob/e2e404305deb52a449fd6449621df2fd77cf0215/LLAFormerNet.png"/> </p>

## Accuracy vs Loss
<p align="center"> <img src="https://github.com/Alamgir-JUST/LLAFormerNet/blob/1ebb8ce6497aba24635341a1c1dae3ec323d9529/ACC-Loss-LLAFormerNet.png"/> </p>

## Acknowledgments

We would like to acknowledge the Skill Morph Research Lab, Skill Morph, Dhaka, Bangladesh, for their valuable support throughout this research.

## Contact
Md. Alamgir Hossain,
MSc in ICT, IICT, BUET; BSc in CSE, JUST. 
Director, Skill Morph Research Lab., Skill Morph, Dhaka, Bangladesh.
Mail: alamgir.cse14.just@gmail.com

Google Scholar: https://scholar.google.com/citations?user=P-_d2XsAAAAJ&hl=en&oi=sra
