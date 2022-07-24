##################################################
# Author: S. A. Owerre
# Date modified: 21/07/2022
# Function: Test for Semi-supervised Transformation
###################################################

import pandas as pd
import numpy as np
import pytest
import sys
base_path = ''
sys.path.append(base_path + 'anomaly-detection/network-intrusion/src/helper/')
import ss_transfxns as tfxn

@pytest.fixture()
def input_df():
    return pd.DataFrame(
        {
            'v_0':np.random.uniform(low=10, high=100, size=10,),
            'v_1':np.random.uniform(low=10, high=100, size=10,),
            'v_2':np.random.uniform(low=10, high=100, size=10,),
            'v_3':np.random.uniform(low=10, high=100, size=10,),
            'v_4':np.random.uniform(low=10, high=100, size=10,),
            'v_5':np.random.uniform(low=10, high=100, size=10,),
            'v_6':np.random.uniform(low=10, high=100, size=10,),
            'v_7':['r', 'g','b','r', 'g','b','r', 'g','b','b',],


        }
    )

def test_ss_transfxn(input_df):
    """Test preprocessing with both 
    numerical and categorical variables.
    """
    model = tfxn.TransformationPipeline()
    X_train_scaled, feat_names = model.preprocessing(input_df)
    num_cols_expected = input_df.shape[1] + len(input_df.v_7.unique())-1
    X_train_scaled.shape == (input_df.shape[0], num_cols_expected)
    len(feat_names) == num_cols_expected