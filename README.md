# Iris Classification Machine Learning Project

## Overview

This project demonstrates a simple machine learning classification workflow using the Iris dataset and a K-Nearest Neighbors (KNN) classifier.

The project was developed using Python and scikit-learn and includes:

- Exploratory work and model development in a Jupyter Notebook
- A reproducible Python training script
- An automated test using pytest
- A confusion matrix generated as a model evaluation output

## Dataset

The project uses the built-in Iris dataset provided by scikit-learn.

The dataset contains measurements of iris flowers from three species:

- Setosa
- Versicolor
- Virginica

The features used for classification are:

- Sepal length
- Sepal width
- Petal length
- Petal width

## Machine Learning Model

The project uses a K-Nearest Neighbors (KNN) classifier with:

```text
n_neighbors = 3
```
## Project Structure
ML_Iris_Project/
├── notebooks/
│   └── iris_model.ipynb
├── src/
│   └── train.py
├── tests/
│   └── test_train.py
├── outputs/
│   └── confusion_matrix.png
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt