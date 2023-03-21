
import data.utils.duckdb_utils as ddu
import duckdb
import data.utils.data_utils as du
from featurewiz import featurewiz
import random
import lightgbm as lgb
import numpy as np
import sklearn.metrics as mt
import pandas as pd
from data.feature_selection_specs.base import base_feature_selection
import gc

class feature_selection(base_feature_selection):
    def __init__(self,
                 master_config_path, 
                 master_config_name,
                 db_connection,
                 database_path=None, 
                 train_feature_table=None,
                 train_feature_selection_table=None,
                 train_feature_info_table=None,
                 filter_out_cols=None,
                 ignore_cols = None,
                 verbose=True):
        base_feature_selection.__init__(self, 
                                        master_config_path, 
                                        master_config_name,
                                        db_connection,
                                        database_path=database_path, 
                                        train_feature_table=train_feature_table,
                                        train_feature_selection_table = train_feature_selection_table,
                                        train_feature_info_table=train_feature_info_table,
                                        filter_out_cols=filter_out_cols,
                                        ignore_cols = ignore_cols,
                                        verbose=verbose)
    
    def set_feature_selection_name(self):
        self.feature_selection_method = 'featurewiz'
        
    def perform_feature_selection(self,df_features,label):
        #print(df_features)
        final_col = [col for col in df_features.columns.tolist() if 'timestamp' not in col]
        df_features = df_features[final_col]
        for col in df_features.columns.tolist():
            print(col)
        ret_obj = featurewiz(df_features,label,corr_limit=0.90,verbose=2)
        selected_features = ret_obj[0]
        ret_dict = {}
        null_cols = du.checknans(df_features, threshold=100)
        selected_features = [col for col in selected_features if col not in null_cols]
        for col in selected_features:
            ret_dict.update({col:0.0})
        del ret_obj
        return ret_dict
