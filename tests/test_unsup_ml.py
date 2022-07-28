#######################################
# Author: S. A. Owerre
# Date modified: 21/07/2022
# Function: Test for Unsupervised ML
########################################

import pandas as pd
import numpy as np
import pytest
import sys
base_path = ''
sys.path.append(base_path + 'anomaly-detection/network-intrusion/src/helper/')
import unsup_ml as ml
import transfxns as tfxn

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
        }
    )
    return df

def test_unsupervised(input_df):
    """Test Unsupervised ML."""
    model = ml.UnsupervisedModels()
    transfxn = tfxn.TransformationPipeline()
    X_scaled, _, _ = transfxn.preprocessing(input_df, input_df)
    # Test Isolation Forest
    if_scores, if_y_pred = model.iforest(
    X_scaled, n_estimators=100, random_state=42
    )
    assert len(if_scores) == X_scaled.shape[0]
    assert np.max(if_scores) <= 100
    assert np.min(if_scores) >= 0
    assert np.array_equal(np.unique(if_y_pred), np.array([0,1])) is True
    assert len(if_y_pred) == X_scaled.shape[0]

    # Test CBLOF model from PYOD
    cblof_scores, cblof_y_pred = model.cblof(
        X_scaled, contamination=0.1, random_state=42
    )
    assert len(cblof_scores) == X_scaled.shape[0]
    assert np.max(cblof_scores) <= 100
    assert np.min(cblof_scores) >= 0
    assert np.array_equal(np.unique(cblof_y_pred), np.array([0,1])) is True
    assert len(cblof_y_pred) == X_scaled.shape[0]

    # Test OCSVM model
    ocsvm_scores, ocsvm_y_pred = model.ocsvm(
        X_scaled, kernel='rbf', gamma=0.1, nu=0.1
    )
    assert len(ocsvm_scores) == X_scaled.shape[0]
    assert np.max(ocsvm_scores) <= 100
    assert np.min(ocsvm_scores) >= 0
    assert np.array_equal(np.unique(ocsvm_y_pred), np.array([0,1])) is True
    assert len(ocsvm_y_pred) == X_scaled.shape[0]

    # Test Elliptic Envelope model
    ellip_scores, ellip_y_pred = model.cov(
        X_scaled, contamination=0.1, random_state=42
        )
    assert len(ellip_scores) == X_scaled.shape[0]
    assert np.max(ellip_scores) <= 100
    assert np.min(ellip_scores) >= 0
    assert np.array_equal(np.unique(ellip_y_pred), np.array([0,1])) is True
    assert len(ellip_y_pred) == X_scaled.shape[0]

    # Test PCA
    pca_scores, pca_y_pred = model.pca(
        X_scaled, n_components=None, contamination=0.1
    )
    assert len(pca_scores) == X_scaled.shape[0]
    assert np.max(pca_scores) <= 100
    assert np.min(pca_scores) >= 0
    assert np.array_equal(np.unique(pca_y_pred), np.array([0,1])) is True
    assert len(pca_y_pred) == X_scaled.shape[0]

    # Test min-max normalization
    arr = np.array(input_df['v_0'])
    scaler = model.min_max_scaler(arr)
    assert len(scaler) == len(arr)
    assert np.max(scaler) <= 100
    assert np.min(scaler) >= 0