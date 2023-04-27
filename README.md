# Sample Constraint - Income Predictor

---

## Project Team
Kaytlynn Skibo, Chris Landschoot, Nicholas Nguyen, Ayako Homma 

---

## Problem Statement 

The goal of this project is to develop a machine learning model that can accurately predict whether an individual's income exceeds $50,000 per year while working within the constraints of a given sample dataset, which represents only 20% of the larger dataset. 
Given the limited size of the sample dataset, the development of an accurate predictive model requires careful feature selection, model development, and testing. 
Completion of this project will offer insights into the model development within constrained sample dataset.

--- 

## Data

This project uses the cheap_train_sample dataset, which represents only 20% of the large_train_sample dataset. This [Adult Data Set](https://archive.ics.uci.edu/ml/datasets/adult) is availabke on UCI Machine Learning Repository website. 

- [cheap_train_sample.csv](/data/original/cheap_train_sample.csv)
- [large_train_sample.csv](/data/original/large_train_sample.csv)
- [test_data.csv](/data/original/test_data.csv)

---

## Exploratory Data Analysis (EDA) 

EDA revealed some interesting findings. First, the percentage breakdown of wages by education level showed that individuals with higher levels of education, such as those with doctorate or professional school degrees, had a higher percentage of shares in wages over $50,000 compared to those with less education, such as those with preschool or some high school education. 
![pct_breakdown of wages_education](/images/pct_breakdown%20of%20wages_education.png)

Second, the percentage breakdown of wages by occupation showed that executive managerial and professional specialty roles had a higher percentage of shares in wages over $50,000, while roles in armed forces and private house services had the lowest percentages. 
![pct_breakdown_wages_occupation](/images/pct_breakdown_wages_occupation.png)

Finally, the percentage breakdown of wages by marital status revealed that individuals who were married-civ-spouse or married-af-spouse had the highest percentage of shares in wages over $50,000. 
![pct_breakdown_wages_maritial status](/images/pct_breakdown_wages_maritial%20status.png)

---

## Modeling

Modeling process involved testing three different classifier models: Logistic Regression, KNN, and SVC. We used GridSearch to find the best parameters for each model. After training the models with the best parameters, we calculated several performance metrics, including accuracy, precision, recall and specificity. Additionally, we evaluated the results using a confusion matrix. 

--- 

## Evaluation

Here are metrics to evaluate different classifier models. 

|Model Type|Train Accuracy|Test Accuracy|Specificity|Precision|Sensitivity|
|--|--|--|--|--|--|
|Logistic Regression|0.823|0.814|0.796|0.797|0.833
|KNN|1.000|0.991|0.984|0.983|0.998|
|SVM|0.875|0.866|0.816|0.827|0.918|

Based on the above table, the KNN model achieved the highest Test Accuracy of 0.991, indicating that it performed the best in predicting whether an individual's income exceeds $50,000 per year. However, since the training data was bootstrapped due to a small dataset, the model may be overfit and may not perform as well on unseen data. 

The ensembling of models was initially attempted but abandoned due to the superior performance of the KNN model. To improve the model's performance, suggestions include testing other models, collecting more data, and engineering new features. In terms of evaluation metrics, the KNN model achieved the highest Specificity, Precision, and Sensitivity values, indicating its superiority over other models. 

---

## Conclusion & Recommendations

The KNN model was the best performing model to predict whether an individual's income exceeds $50,000 per year. We recommend further exploration of the data to identify other features that may impact an individual's income level and refining the model to achieve even better performance metrics. Furthermore, it is important to consider that this model was built using only a sample dataset and may not generalize well to other datasets. Therefore, future studies should explore the application of this model to larger and more diverse datasets to test its generalizability.

---

## Data Dictionary

| Feature | Type | Dataset | Description |
|--|--|--|--|
| age | continuous | Adult Data Set | Age of the individual   |
| workclass | categorical |Adult Data Set| Type of employer for the individual |
| fnlwgt | continuous | Adult Data Set| Final weight used in analysis   |
| education | categorical | Adult Data Set  | Highest level of education completed by the individual |
| education-num | continuous | Adult Data Set | Number of years of education completed |
| marital-status | categorical | Adult Data Set| Marital status of the individual |
| occupation | categorical | Adult Data Set| Type of occupation for the individual |
| relationship | categorical | Adult Data Set | Relationship status of the individual |
| race | categorical | Adult Data Set  | Race of the individual |
| sex | categorical | Adult Data Set   | Gender of the individual |
| capital-gain | continuous | Adult Data Set   | Capital gains for the individual |
| capital-loss | continuous | Adult Data Set    | Capital losses for the individual |
| hours-per-week | continuous | Adult Data Set | Number of hours worked per week |
| native-country | categorical | Adult Data Set  | Country of origin for the individual |