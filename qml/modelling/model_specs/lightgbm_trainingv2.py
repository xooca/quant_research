import sys,os
import qml.data.utils.duckdb_utils as ddu
import duckdb
import qml.data.utils.data_utils as du
import random
import lightgbm as lgb
import numpy as np
import sklearn.metrics as mt
from qml.modelling.model_specs.base import base_model
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
                 train_tuning_info_table=None,
                 ignore_column=None,
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
                                train_tuning_info_table=train_tuning_info_table,
                                ignore_column=ignore_column,
                                verbose=verbose)
        
    
    
    def create_fit_params(self):
        def evaluate_macroF1_lgb(truth, predictions):  
            # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
            pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)
            f1 = mt.f1_score(truth, pred_labels, average='macro')
            return ('macroF1', f1, True) 
        
        self.model_fit_params = {}
        self.model_fit_params.update({'X':self.train_data})
        self.model_fit_params.update({'y':self.train_data_label})
        
        self.model_fit_params.update({'eval_set':[(self.validation_data, self.validation_data_label), 
                                                  (self.train_data, self.train_data_label)]})
        self.model_fit_params.update({'eval_names':['valid','train']})
        if self.add_cat_col_to_model:
            self.model_fit_params.update({'categorical_feature':self.cat_cols})
        self.model_fit_params.update({'eval_metric':evaluate_macroF1_lgb})
        
        print(f"Elements is model_fit_params are {self.model_fit_params.keys()}")
        
    def create_fit_params1(self):
        if self.model_fit_params is None:
            self.model_fit_params = {}
        self.model_fit_params.update({'X':self.train_data})
        self.model_fit_params.update({'y':self.train_data_label})
        
        self.model_fit_params.update({'eval_set':[(self.validation_data, self.validation_data_label), 
                                                  (self.train_data, self.train_data_label)]})
        self.model_fit_params.update({'eval_names':['valid','train']})
        if self.add_cat_col_to_model:
            self.model_fit_params.update({'categorical_feature':self.cat_cols})
        print(f"Elements is model_fit_params are {self.model_fit_params.keys()}")

    def model_spec_info(self):
        self.algo_name = 'lightgbm'
        self.tuning_type = 'optuna'
        self.convert_to_cat = False
        self.add_cat_col_to_model = False
        self.if_return_df = True
        self.train_data_scamble=True
        self.train_sampling_fraction=1
        self.validation_data_scamble=True
        self.validation_sampling_fraction=1
        self.reduce_data_size = False
        self.scramble_all=True
        self.comb_diff=3
        self.select_value=4
        self.stride=2
        self.strategy='chunking'
        self.feature_strategy = 'selection' # 'all-notinclude-unstable-feature','all-include-unstable-feature','selection'
        self.standardize_strategy = None # 'all','only-unstable'
              
    def create_model(self):
        model = None
        print(f"Model parameters are : {self.model_params}")
        self.model_params.update({'is_unbalance':True})
        model = lgb.LGBMClassifier(**self.model_params)
        return model