# Plastic Prediction Demo - Simple ML Pipeline

This repository contains a Python script demonstrating a basic machine learning workflow using synthetic data.

## About

The script generates a synthetic dataset for a binary classification problem, trains a Random Forest classifier, evaluates its performance, and shows feature importance. This serves as a simple example of how to build, train, and evaluate a machine learning model without needing any external data.

## How It Works

1. **Data Generation**  
   Creates fake data with 500 samples and 10 features for classification.

2. **Data Splitting**  
   Splits data into training (75%) and testing (25%) sets.

3. **Model Training**  
   Trains a Random Forest classifier on the training data.

4. **Evaluation**  
   Prints accuracy and a detailed classification report on test data.

5. **Feature Importance**  
   Displays the importance of each feature in the model’s decisions.

## Usage

To run the script, make sure you have the required Python packages installed:

```bash
pip install scikit-learn numpy
# ML_pipeline
