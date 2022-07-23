#######################################
# Author: S. A. Owerre
# Date modified: 21/07/2022
# Function: Test for Log Transformation
########################################

import pandas as pd
import pytest
import sys
base_path = '/Users/sowerre/Documents/python/ml-projects/'
sys.path.append(base_path + 'anomaly-detection/network-intrusion/src/')
import LogTransformer as lg