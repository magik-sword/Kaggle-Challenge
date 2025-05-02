![](UTA-DataScience-Logo.png)

# Predict Health Outcomes of Horses

This repository holds files to build a XGBoost model to predict health outcomes from "Predict Health Outcomes of Horses" Kaggle challenge (https://www.kaggle.com/competitions/playground-series-s3e22) 

## Overview

  * Challenge: The task, as defined by the Kaggle challenge is to use a a dataset of health indicators to predict the health outcome of horses.
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
* Leave outliers in the dataset
* Imputed missing values in categorical columns with mode. (Numerical values had no missing values)
* Save a copy of the 'id' column from test.csv to use later for the Kaggle submission
* Remove redundant/irrelevant columns: 'id', 'hospital_number', 'cp_data', 'lesion_2', 'lesion_3'
* One-hot encode all the categorical features except for the outcome column

#### Data Visualization

![image](https://github.com/user-attachments/assets/a9a3a618-1bcd-4374-ba22-255d89564d2b)

Features such as nasogastric_reflux_ph and abdomo_protein were highly useful because the euthanized class was clearly seperated from both the living and died classes. The nasogastric_reflux_ph for euthanized horses is outside the normal range, but strangely abdimi_protein is low, which indicates a decreased chance of a compromised gut.

![image](https://github.com/user-attachments/assets/b63215d8-84b8-4235-b936-85a81e7ec1e9)

The barplot shows there's a moderate class imbalance in the dataset, something to be taken into account when model training began.


### Problem Formulation

* Define:
  * Input: train.csv
  * Output: submission.csv
  * Model
    * XGBoost: Robust to outliers, no scaling needed, and strong multi-class classification performance.
  * Tuned Hyperparameters: objective='multi:softmax', random_state=42, gamma=0.3, learning_rate=0.1, max_delta_step=3, max_depth=3, n_estimators=200)

### Training

* The training process is straightforward: Load the train, test, and submission CSV files and label-encode the outcome column as integers. Split the training data into training and validation sets, and train a baseline XGBoost model. Then perform hyperparameter tuning using GridSearchCV (feel free to experiment with different hyperparameters). Finally, initialize and fit XGBoost using the optimized parameters, reverse the label encoding on the predictions, and generate outcome predictions on the cleaned test dataset for submission. I decided to stop training here because whenever I tried to add more hyperparameters or values to try, the run time was just too long. 

### Performance Comparison

* ALthough I did look at precision, recall, and F1 for each class given this was an imbalanced dataset, the key metric was micro-averaged F1 score. This metric sums up all of the TP, FN, and FP across all classes and then applies the F1 score formula. This is useful for multi-class classification because it gives equal weight to each class and essentially acts like accuracy despite there being a class imbalance.
![image](https://github.com/user-attachments/assets/ba2e2942-bc30-4ea4-af2b-06b5ce593f0a)
![image](https://github.com/user-attachments/assets/d8f3626e-4f5e-4c25-8771-bf44c56440b5)



### Conclusions

![image](https://github.com/user-attachments/assets/a871120d-4dd9-4934-b2f4-2d9d029dc283)

* XGBoost, like many other tree-based models, perform excellent on multi-classification problems, even when given relatively little data to work with.

### Future Work

* I would go back and see if scaling or handling outliers in a different way would've given better results.
* I didn't try it during this challenge, but the description on the Kaggle page suggested the idea of incorporating the original, real-world dataset when training the model. It would be interesting to see how this would affect my model.
  
## How to reproduce results

* Reproducing the results in these files should be easy. By running all the cells in 'Data Visualization and Pre-Processing', one should receive a cleaned test.csv, train.csv, and a submission.csv with only the test_ids. Running all of the cells once again in the 'Training and Evaluation' file should result in a proper submission.csv with the test_ids in the first column, and the predictions for each entry in the second. If for some reason something doesn't work, I briefly described the pre-processing steps I performed at the beginning of this file and both files have lots of documentation about what I did.

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

#### Performance Evaluation

* There is a function in Training and Evaluation that will display classification report along with the micro-averaged F1 score. 

## Citations

* Why do tree-based models still outperform deep learning on tabular data?: (https://doi.org/10.48550/arXiv.2207.08815) 
