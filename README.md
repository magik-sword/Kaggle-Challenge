![](UTA-DataScience-Logo.png)

# Predict Health Outcomes of Horses

This repository holds files to build a XGBoost model to predict health outcomes from the "Predict Health Outcomes of Horses" Kaggle challenge (https://www.kaggle.com/competitions/playground-series-s3e22). 

## Overview

  * Challenge: The task, as defined by the Kaggle challenge is to use a a dataset of health indicators such as pulse, respiratory rate, and temperature to predict whether a horse lived, died, or was euthanized. 
  * Approach: The challenge is a multi-class classification one, so a XGBoost model with tuned hyperparameters was chosen.
  * Summary: The XGBoost model achieved a micro-averaged F1 score of 0.73484, the current best score for the challenge is 0.78181.
    
## Summary of Workdone

### Data

  * Type: Categorical and Numerical
    * Input: train.csv: Health indicators with outcome column at the end
    * Input: test.csv: Health indicators with outcome column removed
    * Input: sample_submission.csv: Example of predicted outcomes of the test set to send for evaluation
  * Size:
    * train: 1,235 rows, 29 features, size: 229.2 kB
    * test: 824 rows, 28 features, size: 148.32 kB
    * sample_submission: 824 rows, 2 features, size: 9.07 kB
  * Instances (Train, Test, Validation Split): 988 training, 247 validation, and 824 test 

#### Preprocessing / Clean up
All of the steps below were applied to test.csv and train.csv to maintain dataset alignment:
* Converted lesion_1, lesion_2, lesion_3 to categorical features to keep data types consistent
* Kept outliers in both datasets to preserve potentially valuable information
* Imputed missing values in categorical columns using mode from train.csv; no missing values were found in numerical features
* Remove redundant or irrelevant columns: 'id', 'hospital_number', 'cp_data', 'lesion_2', 'lesion_3'
* One-hot encoded the remaining categorical features, except the target column 'outcome' in train.csv

#### Data Visualization

![image](https://github.com/user-attachments/assets/a9a3a618-1bcd-4374-ba22-255d89564d2b)

Features such as 'nasogastric_reflux_ph' and 'abdomo_protein' were highly useful for the model because the euthanized class was clearly separated from both the living and died classes. 

![image](https://github.com/user-attachments/assets/bcab9bb3-6dd7-416e-b7a3-12fdf848007a)

In contrast, features like 'lesion_3' were irrelevant. Over 99% of the samples in each class had no lesion, meaning this feature didn't effectively differentiate between the classes.


### Problem Formulation

* Define:
  * Input: train.csv, test.csv, sample_submission.csv
  * Output: submission.csv
  * Model
    * XGBoost: Robust to outliers, no scaling needed, and strong multi-class classification performance.
  * Tuned Hyperparameters: objective='multi:softmax', random_state=42, gamma=0.3, learning_rate=0.1, max_delta_step=3, max_depth=3, n_estimators=200)

### Training

* Load the train, test, and submission CSV files generated from 'Data Processing and Pre-Processing' file as Pandas dataframes
* Label-encode the outcome column
* Split the training data into training and validation sets, and train a baseline XGBoost model.
* Perform hyperparameter tuning using GridSearch.
* Evaluate a new XGBoost model using the optimized parameters
* Fit this new model to the test set, reverse the label encoding on the predictions, and append them to the submission dataframe
* Create the submission file to send to Kaggle for evaluation
  
### Performance Comparison

![image](https://github.com/user-attachments/assets/ba2e2942-bc30-4ea4-af2b-06b5ce593f0a)
![image](https://github.com/user-attachments/assets/d8f3626e-4f5e-4c25-8771-bf44c56440b5)

Although I did look at precision, recall, and F1 to gauge individual class performance, the key metric I focused on improving was micro-averaged F1 score

### Conclusions

* XGBoost performs excellent on multi-classification problems, even when given relatively little data to work with.

### Future Work

* Try scaling or handling outliers in a different way
* Remove highly correlated features
* Incorporate the original, real-world dataset into training the model
  
## How to reproduce results

Reproducing the results in these files should be easy. By running all the cells in 'Data Visualization and Pre-Processing', one should receive a cleaned test.csv, train.csv, and a submission.csv with only the test_ids. Running all of the cells once again in the 'Training and Evaluation' file should result in a proper submission.csv with the test_ids in the first column, and the predictions for each entry in the second. If for some reason something doesn't work, check the code files for documentation.

### Overview of files in repository

  * Data Visualizations and Pre-Processing.ipynb: Exploratory data analysis, visualizations, and preprocessing steps. Creates a processed train, test, and partial submission file.
  * Training and Evaluation.ipynb: Trains and evaluates the model. Outputs the submission.csv with a (id, outcome) format.
  * submission.csv: id and predicted outcome of each entry in the test set

### Required Libraries
* Pandas
* Numpy
* Matplotlib
* IPython
* Tabulate
* Sci-kit 
* XGBoost

#### Performance Evaluation

There is a function in Training and Evaluation that will display classification report along with the micro-averaged F1 score for the model when called.

## Citations

Why do tree-based models still outperform deep learning on tabular data?: (https://doi.org/10.48550/arXiv.2207.08815) 
