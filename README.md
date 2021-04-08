# Anomaly Detection of CreditCard Fraud & Network Intrusion

Anomaly detection falls into two main categories:

## 1. Outlier Detection:
  In outlier detection, the training data contains outliers which are defined as observations that are far from the others. The objective is to detect the outliers in a new observation.

## 2. Novelty Detection:
 In	novelty detection, a semi-supervised learning technique, the training data is not polluted by outliers. It is trained to learn the high and low density regions in the feature space, and we are interested in detecting whether a new observation is an outlier.

## Exploratory Data Analysis
We show the result of the PCA dimensionality reductions of the two datasets.

![PCA Plot1](creditcard/image/pca.png)

![PCA Plot2](network_intrusion/image/pca.png)

## Methods
To study the anomaly detection of credit card fraud & network intrusion, we trained and evaluated the following unsupervised learning techniques

1). Local Outlier Factor

2). One-Class SVM

3). Isolation Forest

4). Elliptic Envelope

Furthermore, we also trained the imbalanced datasets using supervised learning methods such as

a). Logistic Regression Classifier

b). Random Forest Classifier

c). XGBoost Classifier
