#######################################
# Author: S. A. Owerre
# Date modified: 21/07/2022
# Function: Test for Semi-supervised ML
########################################

import pandas as pd
import numpy as np
import pytest
import sys
base_path = ''
sys.path.append(base_path + 'anomaly-detection/network-intrusion/src/helper/')
from sklearn.linear_model import LogisticRegression
import ss_ml as ml
import ss_transfxns as tfxn

@pytest.fixture()
def input_df():
    df = pd.DataFrame(
        {
            'v_0':np.random.uniform(low=10, high=100, size=50,),
            'v_1':np.random.uniform(low=10, high=100, size=50,),
            'v_2':np.random.uniform(low=10, high=100, size=50,),
            'v_3':np.random.uniform(low=10, high=100, size=50,),
            'v_4':np.random.uniform(low=10, high=100, size=50,),
            'v_5':np.random.uniform(low=10, high=100, size=50,),
            'v_6':np.random.uniform(low=10, high=100, size=50,),
            'class':np.random.randint(3, size = 50,),
        }
    )
    x, y = df.drop('class', axis=1), df['class']
    y_ = y.map({0:0, 1:1, 2:-1})
    return x, y_

def test_semi_supervised(input_df):
    """Test semi-supervised ML."""
    model = ml.SemiSupervised()
    transf = tfxn.TransformationPipeline()
    X_train, y_train = input_df
    X_train_scaled, _ = transf.preprocessing(X_train)
    base_classifier = LogisticRegression(random_state=42)
    y_pred, y_proba = model.self_training_clf(
        base_classifier, 
        X_train_scaled, 
        y_train,
        threshold=0.75, 
        max_iter=None,
        verbose=True
    )
    assert len(y_pred) == len(y_train)
    assert np.array_equal(np.unique(y_pred), np.array([0,1])) is True
    assert np.max(y_proba) <= 1