import sys,os
import data.utils.duckdb_utils as ddu
import duckdb
import data.utils.data_utils as du
import random
import lightgbm as lgb
import numpy as np
import sklearn.metrics as mt
from modelling.model_specs.base_v2 import base_model
import pickle
import pandas as pd
import pickle
import pandas as pd
from collections import Counter
import gc
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor,CatBoostClassifier
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score
from sklearn.model_selection import train_test_split
import pickle
from scipy.special import expit
import datetime as dt

import sys,os
import data.utils.duckdb_utils as ddu
import duckdb
import data.utils.data_utils as du
import random
import numpy as np
import sklearn.metrics as mt

import pickle
import pandas as pd
import pickle
import pandas as pd
from collections import Counter
import gc
from modelling.metrics import metrics as met

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
                 ignore_column = None,
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
                
        self.model_fit_params = {}
        self.model_fit_params.update({'X':self.train_data})
        self.model_fit_params.update({'y':self.train_data_label})
        
        #self.model_fit_params.update({'eval_set':[(self.validation_data, self.validation_data_label), 
        #                                        (self.train_data, self.train_data_label)]})
        self.model_fit_params.update({'eval_set':[(self.validation_data, self.validation_data_label)]})

        #self.model_fit_params.update({'categorical_feature':self.cat_cols})            
        print(f"Elements is model_fit_params are {self.model_fit_params.keys()}")
        

    def model_spec_info(self):
        self.algo_name = 'catboost'
        self.tuning_type = 'optuna'
        self.convert_to_cat = True
        self.add_cat_col_to_model = True
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
        #self.feature_strategy = 'all-include-unstable-feature' # 'all-notinclude-unstable-feature','all-include-unstable-feature','selection'
        self.feature_strategy = 'selection'

        self.standardize_strategy = None # 'all','only-unstable'
        self.save_extra_info_flag = True
    
        
    def save_extra_info(self,model):
        feature_imp= pd.DataFrame({'feature_importance': model.get_feature_importance(), 
                                   'feature_names': self.feature_names}).sort_values(by=['feature_importance'], 
                                                                                      ascending=False)
        print("FEATURE IMPORTANCE IS")
        print(feature_imp)
        curr_dt = str(dt.datetime.now()).replace(".","").replace("-","")
        f_imp_path = f"{self.train_model_base_path}train_catboost_feature_imp_{curr_dt}.csv"
        feature_imp.to_csv(f_imp_path)
        model_path = f"{self.train_model_base_path}train_catboost_model_{curr_dt}.pkl"
        du.save_object(model_path,model)
        
    def create_model(self):
        if self.convert_to_cat:
            self.model_params.update({'cat_features':self.cat_cols})
        print(f"Model Parameter is {self.model_params}")
        model = CatBoostClassifier(**self.model_params)
        #self.create_fit_params()
        #callback_func = str(param['eval_metric']).split('object')[0].strip().split(".")[-1]
        #self.model_fit_params.update({'callbacks':[CatBoostPruningCallback(trial,callback_func)]})
        #model.fit(**self.model_fit_params)

        #feature_imp= pd.DataFrame({'feature_importance': model.get_feature_importance(), 
        #                           'feature_names': self.feature_names}).sort_values(by=['feature_importance'], 
        #                                                                              ascending=False)
        #print("FEATURE IMPORTANCE IS")
        #print(feature_imp)
        #curr_dt = str(dt.datetime.now()).replace(".","").replace("-","")
        #f_imp_path = f"{self.train_model_base_path}train_feature_imp_{curr_dt}.csv"
        ##feature_imp.to_csv(f_imp_path)
        #model_path = f"{self.train_model_base_path}train_model_{curr_dt}.pkl"
        #du.save_object(model_path,model)
        return model
    
    def create_model2(self):
        if self.convert_to_cat:
            self.params.update({'cat_features':self.cat_cols})
        print(f"Model Parameter is {self.model_params}")
        model = CatBoostClassifier(**self.model_params)
        #self.create_fit_params()
        #self.model_fit_params.update({'callbacks':[CatBoostPruningCallback(trial, ProfitMetric(),0)]})
        #print(f"Fit Parameter is {self.model_fit_params}")
        #model.fit(**self.model_fit_params)
        return model