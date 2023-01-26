
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
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import sys
import joblib

sys.modules['sklearn.externals.joblib'] = joblib

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
                                        verbose=verbose)
    
    def correlation(self,df_features, threshold):
        col_corr = set()  # Set of all the names of correlated columns
        corr_matrix = df_features.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
        return col_corr
    
    def set_feature_selection_name(self):
        self.feature_selection_method = 'mlextend_sfs'
        
    def perform_feature_selection(self,df_features,label):
        labels = df_features[label].tolist()
        max_label =max(labels,key=labels.count)
        df_features = df_features[df_features[label]==max_label]
        cols = [col for col in df_features.columns if col != label]
        corr_features = self.correlation(df_features[cols], 0.8)
        print(f"Corelated features are {corr_features}")
        print(f"Before cor drop shape is {df_features.shape}")
        df_features.drop(labels=corr_features, axis=1, inplace=True)
        print(f"After cor drop shape is {df_features.shape}")

        sfs = SFS(RandomForestClassifier(n_estimators=75, n_jobs=4, random_state=0),
                k_features=150, # the lower the features we want, the longer this will take
                forward=False,
                floating=False,
                verbose=2,
                scoring='accuracy',
                cv=2)
        
        sfs = sfs.fit(df_features[cols], df_features[label])
        selected_features= list(sfs.k_feature_names_)

        return selected_features
