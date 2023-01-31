import sys,os
import data.utils.duckdb_utils as ddu
import duckdb
import data.utils.data_utils as du
import random
import lightgbm as lgb
import numpy as np
import sklearn.metrics as mt
from modelling.model_specs.base import base_model
import pickle
import pandas as pd
import pickle
import pandas as pd
from collections import Counter
import gc

class model(base_model):
    def __init__(self,
                 master_config_path, 
                 master_config_name,
                 db_connection,
                 database_path=None, 
                 train_feature_table = None,
                 train_feature_selection_table=None,
                 train_feature_info_table=None,
                 train_training_info_table=None,
                 verbose=True):
        base_model.__init__(self, 
                                master_config_path=master_config_path, 
                                master_config_name=master_config_name,
                                db_connection=db_connection,
                                database_path=database_path, 
                                train_feature_table = train_feature_table,
                                train_feature_selection_table=train_feature_selection_table,
                                train_feature_info_table=train_feature_info_table,
                                train_training_info_table=train_training_info_table,
                                verbose=verbose)
        
    
    def model_spec_info(self):
        self.algo_name = 'lightgbm'
        self.tuning_type = 'optuna'
           
    def create_model(self,model_params):
        model = None
        print(f"Model parameters are : {model_params}")
        model = lgb.LGBMClassifier(**model_params)
        return model