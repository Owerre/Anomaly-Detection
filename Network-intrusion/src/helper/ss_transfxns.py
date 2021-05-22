# Filter warnings
import warnings
warnings.filterwarnings("ignore")

# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data pre-processing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Helps to import functions from another directory
import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from helper import log_transfxn as cf 

# Dimensionality reduction
from sklearn.decomposition import PCA

class TransformationPipeline:
    """
    A class for transformation pipeline for semi-supervised learning
    """
    def __init__(self):
        """
        Define parameters
        """
        
    def num_pipeline(self, X_train):
        """
        Transformation pipeline of data with only numerical variables

        Parameters
        ___________
        X_train: Training set

        Returns
        __________
        Transformation pipeline and transformed data in numpy array
        """
         # Original numerical feature names 
        feat_names = list(X_train.select_dtypes('number'))

        # Create pipeline
        num_pipeline = Pipeline([
                                 ('std_scaler', StandardScaler()),
                                ])

        # Apply transformer
        X_train_scaled = num_pipeline.fit_transform(X_train)
        return X_train_scaled, feat_names
    
    def cat_encoder(self, X_train):
        """
        Encoder for categorical variables

        Parameters
        ___________
        X_train: Training setcd

        Returns
        __________
        Transformation pipeline and transformed data in array
        """
        # Instatiate class
        one_hot_encoder = OneHotEncoder()

        # Fit transform the training set
        X_train_scaled = one_hot_encoder.fit_transform(X_train)
        
        # Feature names for output features
        feat_names = list(one_hot_encoder.get_feature_names(list(X_train.select_dtypes('O'))))
        return X_train_scaled.toarray(), feat_names

    def preprocessing(self, X_train):
        """
        Transformation pipeline of data with both numerical and categorical variables.

        Parameters
        ___________
        X_train: Training set

        Returns
        __________
        Transformed data in array
        """

        # Numerical transformation pipepline
        num_train, num_col = self.num_pipeline(X_train.select_dtypes('number'))

        # Categorical transformation pipepline
        cat_train, cat_col = self.cat_encoder(X_train.select_dtypes('O'))

        # Transformed training set
        X_train_scaled = np.concatenate((num_train,cat_train), axis = 1)

        # Feature names
        feat_names = num_col + cat_col
        return X_train_scaled, feat_names
    
    def pca_plot_labeled(self, data_, labels, palette = None, ax = None):
        """
        Dimensionality reduction of labeled data using PCA 

        Parameters
        __________
        data: transformed and scaled data
        labels: class labels
        palette: color list
        ax : matplotlib axes

        Returns
        __________
        Matplotlib plot of two component PCA
        """
        #PCA
        pca = PCA(n_components = 2)
        X_pca = pca.fit_transform(data_)

        # put in dataframe
        X_reduced_pca = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
        X_reduced_pca['class'] = labels

        # plot results
        sns.scatterplot(x = 'PC1', y = 'PC2', data = X_reduced_pca, hue = 'class', 
                        style = 'class',palette = palette, ax = ax)

        # axis labels
        ax.set_xlabel("Principal component 1")
        ax.set_ylabel("Principal component 2")
        ax.legend(loc = 'best')
    
    def plot_pca(self, X_train, y_train, y_pred):
        """
        Plot PCA before and after semi-supervised classification

        Parameters
        ___________
        X_train: Scaled feature matrix of the training set
        y_train: Original labels of the training set
        y_pred: Predicted labels of the unlabeled data points

        Returns
        _____________
        Matplotlib figure
        """    
        # Plot figure
        plt.rcParams.update({'font.size': 15})
        fig, (ax1,ax2) = plt.subplots(1,2, figsize = (20,6))

        self.pca_plot_labeled(X_train, y_train, palette = ['lime', 'r', 'gray'], ax = ax1)
        self.pca_plot_labeled(X_train, y_pred, palette = ['lime', 'r'], ax = ax2)
        ax1.set_title("PCA before semi-supervised classification")
        ax2.set_title("PCA after semi-supervised classification")

