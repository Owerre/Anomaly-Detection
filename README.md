# Anomaly Detection of Network Intrusion

## Data Information

The dataset can be found in  [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OPQMVF)

## Unsupervised Learning

In the unsupervised setting, the class labels of the training set are not available. In the
current problem, the true labels were ignored during training in order to reflect a real-world scenario. Hence, the unsupervised classification models were used to predict the true labels for each record. We trained the following unsupervised models:

1. Isolation Forest
2. Cluster-Based Local Outlier Factor (CBLOF)
3. Principal Component Analysis (PCA)
4. Elliptic Envelope.

In the real-world unsupervised problems, the business have to validate the predicted results due to absence of ground truth. However, in the present problem the predicted labels were validated with the true labels and the results below show that the unsupervised models predicted so many fasle positives.

![fig2](Network-intrusion/image/unsup.png)

## Semi-Supervised Learning

In the semi-supervised setting, a large unlabeled dataset and a small labeled dataset are given. The goal is to train a classifier on the entire dataset that would predict the labels of the unlabeled data points in the training set. This is called transductive semi-supervised learning. In the present problem, we have created a semi-supervised learning dataset consisting of 92\%  unlabeled data points and 8\% labeled data points.

Using self-training semi-supervised learning method, we trained two base classifiers:

1. Logistic Regression
2. Random Forest

We use the ground truth (true lables) of the unlabeled dataset to validate the performance of the self-training semi-supervised learning models, but in reality the ground truth of the unlabeled data points will not be provided. The results are shown below

![fig3](Network-intrusion/image/ss.png)

## Supervised Learning

In the supervised setting, the class label for each record in the training set are provided and the goal is to train a classifier that would be used for the prediction on unseen data. Here, we have trained two classifiers

1. Logistic Regression
2. Random Forest

The results are below show that the two classifiers perform extremely well on the dataset. The AUC-ROC and AUC-PRC are 100\% for on the training (cross-validation) and test sets

![fig4](Network-intrusion/image/sup.png)
