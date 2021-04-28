# Filter warnings
import warnings
warnings.filterwarnings("ignore")

# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Data pre-processing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Anomaly detection models from PYOD
#from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA as PCAOD

# Anomaly detection models from Sklearn
from sklearn.svm import OneClassSVM as OCSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Model performance metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, auc, recall_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve

class UnsupervisedModels:
    """
    Class for training several unsupervised models
    """

    def __init__(self):
        """
        Parameter initialization
        """

    def iforest(self, X_train, n_estimators=None, random_state=None):
        """
        Train Isolation Forest from scikit-learn

        Parameters
        __________
        X_train: scaled training data
        n_estimators: number of isolation trees
        random_state: random number seed

        Returns
        ________
        Anomaly scores
        """
        model  = IsolationForest(n_estimators = n_estimators, max_samples='auto',
                                   random_state = random_state)
        model.fit(X_train)

        # Predict raw anomaly score
        labels = model.predict(X_train) # -1 for outliers and 1 for inliers
        labels = (labels.max() - labels)//2 # rescaled labels (1: outliers, 0: inliers)
        iforest_anomaly_scores = model.decision_function(X_train)*-1 # anomaly score
        iforest_anomaly_scores = self.min_max_scaler(iforest_anomaly_scores)
        return iforest_anomaly_scores, labels

    def cblof(self, X_train, contamination = None, random_state = None):
        """
        Train CBLOF model from PYOD

        Parameters
        __________
        X_train: scaled training data
        contamination: percentage of anomalies in the data
        random_state: random number seed

        Returns
        ________
        Anomaly scores
        """
        model = CBLOF(contamination = contamination, random_state = random_state)
        model.fit(X_train)

        # Predict raw anomaly score
        labels = model.predict(X_train)  # outlier labels (0 or 1)
        cblof_anomaly_scores = model.decision_function(X_train)  # outlier scores
        cblof_anomaly_scores = self.min_max_scaler(cblof_anomaly_scores)
        return cblof_anomaly_scores, labels

    def ocsvm(self, X_train, kernel = None, gamma=None, nu = None):
        """
        Train OCSVM model from Sklearn

        Parameters
        __________
        X_train: scaled training data
        kernel: kernel funcs: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
        gamma: kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
        nu: regularization parameter btw [0,1]

        Returns
        ________
        Anomaly scores
        """
        model  = OCSVM(kernel = kernel, gamma=gamma, nu = nu)
        model.fit(X_train)

        # Predict raw anomaly score
        labels = model.predict(X_train)  # Outlier labels (-1 or 1)
        labels = (labels.max() - labels)//2 # rescaled labels (1: outliers, 0: inliers)
        ocsvm_anomaly_scores = model.decision_function(X_train)*-1  # Outlier scores
        ocsvm_anomaly_scores = self.min_max_scaler(ocsvm_anomaly_scores)
        return ocsvm_anomaly_scores, labels

    def cov(self, X_train, contamination = None, random_state = None):
        """
        Train Elliptic Envelope model from scikit-learn

        Parameters
        __________
        X_train: scaled training data
        contamination: percentage of anomalies in the data
        random_state: random number seed

        Returns
        ________
        Anomaly scores
        """
        model  = EllipticEnvelope(contamination = contamination, random_state = random_state)
        model.fit(X_train)

        # Predict raw anomaly score
        labels = model.predict(X_train) # -1 for outliers and 1 for inliers
        labels = (labels.max() - labels)//2 # rescaled labels (1: outliers, 0: inliers)
        cov_anomaly_scores = model.decision_function(X_train)*-1 # anomaly score
        cov_anomaly_scores = self.min_max_scaler(cov_anomaly_scores)
        return cov_anomaly_scores, labels

    def pca(self, X_train, n_components=None, contamination=None):
        """
        Train PCA model from PYOD

        Parameters
        __________
        X_train: scaled training data
        contamination: percentage of anomalies in the data
        n_components: number of components to transform

        Returns
        ________
        Anomaly scores
        """
        model = PCAOD(n_components= n_components, contamination=contamination)
        model.fit(X_train)

        # Predict raw anomaly score
        labels = model.predict(X_train)  # outlier labels (0 or 1)
        pca_anomaly_scores = model.decision_function(X_train)  # outlier scores
        pca_anomaly_scores = self.min_max_scaler(pca_anomaly_scores)
        return pca_anomaly_scores, labels

    def eval_metric(self, y_true, y_pred, model_nm = None):
        """
         Evaluation metric using the ground truth and the predicted labels

        Parameters
        ___________
        y_pred: predicted labels
        y_true: true labels
        model_nm: name of model

        Returns
        _____________
        Performance metrics
        """
        print('Test predictions for {}'.format(str(model_nm)))
        print('-' * 60)
        print('Accuracy:  %f' % (accuracy_score(y_true, y_pred)))
        print('AUROC: %f' % (roc_auc_score(y_true, y_pred)))
        print('AUPRC: %f' % (average_precision_score(y_true, y_pred)))
        print('Predicted classes:', np.unique(y_pred))
        print('Confusion matrix:\n', confusion_matrix(y_true, y_pred))
        print('Classification report:\n', classification_report(y_true, y_pred))
        print('-' * 60)

    def min_max_scaler(self, arr):
        """
        Min-Max normalization to rescale the anomaly scores

        Parameters
        __________
        arr: 1D array

        Returns
        ________
        normalized array in the range [0,100]
        """
        scaler = (arr-np.min(arr))*100/(np.max(arr)-np.min(arr))
        return scaler

    def plot_dist(self, scores, color = None, title = None):
        """
        Plot the distribution of anomaly scores

        Parameters
        __________
        scores: scaled anomaly scores

        Returns
        ________
        seaborn distribution plot
        """
        # Figure layout
        plt.rcParams.update({'font.size': 15})
        plt.subplots(figsize = (8,6))

        # Plot distribution with seaborn
        sns.distplot(scores, color = color)
        plt.title(label = title)
        plt.xlabel('Normalized anomaly scores')
        plt.show()
