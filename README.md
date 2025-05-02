![](UTA-DataScience-Logo.png)

# Predict Health Outcomes of Horses

This repository holds files to build a XGBoost model to predict health outcomes from "Predict Health Outcomes of Horses" Kaggle challenge (https://www.kaggle.com/competitions/playground-series-s3e22) 

## Overview

  * Challenge: The task, as defined by the Kaggle challenge is to use a a dataset of health indicators to predict the health outcome of a horse.
  * Approach: The approach in this repository formulates the problem as a multi-class classification task, using an optimized XGBoost on one-hot encoded features to predict whether a horse lived, died, or was euthanized.
  * Summary: The XGBoost model achieved a micro-averaged F1 score of 73.48% on the Kaggle test dataset.
## Summary of Workdone

### Data

* Data:
  * Type: Categorical and Numerical
    * Input: train.csv: dataset of health indicators with outcome labeled
  * Size: 1,235 rows, 29 features, file size: 229 kB
  * Instances (Train, Test, Validation Split): 988 training instances, 247 validation instances, 824 test instances

#### Preprocessing / Clean up

* Converted lesion_1, lesion_2, lesion_3 to categorical features
* Imputed missing values in categorical columns with mode
* Save a copy of the 'id' column from test.csv to use later for the Kaggle submission
* Remove 'id', 'hospital_number', 'cp_data', 'lesion_2', 'lesion_3'
* One-hot encode all the categorical features except for the outcome column

#### Data Visualization

![image](https://github.com/user-attachments/assets/a9a3a618-1bcd-4374-ba22-255d89564d2b)
Numerical features such as nasogastric_reflux_ph and abdomo_protein were highly useful because the euthanized class clustered almost entirely in 1 bin


### Problem Formulation

* Define:
  * Input: train.csv
  * Output: submission.csv
  * Model
    * XGBoost: robust to outliers, no scaling needed, and strong multi-class classification performance.
  * Tuned Hyperparameters: objective='multi:softmax', random_state=42, gamma=0.3, learning_rate=0.1, max_delta_step=3, max_depth=3, n_estimators=200)

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

  * Data Visualizations and Pre-Processing.ipynb: Exploratory data analysis, visualizations, and preprocessing steps to get a processed train, test, and submission file.
  * Training and Evaluation.ipynb: Contains code on how to train and evaluate the model. Also contains how to the create the submission.csv file.
  * submission.csv: id and predicted outcome of each entry in the test set

### Software Setup
* import pandas as pd
* import numpy as np
* import matplotlib.pyplot as plt
* from IPython.display import HTML, display
* import tabulate
* from sklearn.preprocessing import OneHotEncoder
* from sklearn.preprocessing import LabelEncoder
* from sklearn.model_selection import train_test_split
* from sklearn.metrics import classification_report, f1_score
* from xgboost import XGBClassifier

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* There is a function in Training and Evaluation that will display classification report along with the micro-averaged F1 score. 

## Citations

* Why do tree-based models still outperform deep learning on tabular data?: (https://doi.org/10.48550/arXiv.2207.08815)








