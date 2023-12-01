# assigment_2PMLDL
# Movie Recommendation System using LightGCN
# Introduction
This repository contains the implementation of a movie recommendation system using the LightGCN collaborative filtering model. The system is trained on the MovieLens dataset, consisting of user ratings for various movies. The goal is to provide personalized movie recommendations based on user preferences.

## Data Analysis
The preprocessing.py script loads and preprocesses the MovieLens dataset, creating mappings for users and movies. The ratings are filtered to include only high ratings, and a histogram of ratings is plotted for data exploration.
![image](https://github.com/mpvnlv/assigment_2PMLDL/assets/88908152/3756d684-a826-4d6f-865d-aee8ee87e434)
## Model Implementation
The core of the recommendation system is the LightGCN model, implemented in the lightgcn.py file. The model is trained using Bayesian Personalized Ranking (BPR) loss, which enables it to learn user and item embeddings for collaborative filtering.

## Training Process
The training process is orchestrated in the train.py script. It involves mini-batch sampling, forward propagation, loss computation, and optimization. The model is trained over multiple epochs, and a learning rate scheduler is applied to enhance training stability.

## Evaluation
Model evaluation is performed on a separate test set using metrics such as recall and precision. The evaluate.py script contains the evaluation logic, and the results provide insights into the model's ability to make accurate recommendations.

## Results
The final results are displayed in the form of training and validation loss plots over epochs, showcasing the model's learning progress. The evaluation metrics on the test set indicate the system's effectiveness in providing personalized movie recommendations.

## How to Use
The entire project is structured within a Colab notebook. Simply clone the repository, download it to your Google Drive, and execute the notebook in Google Colab for seamless usage.





