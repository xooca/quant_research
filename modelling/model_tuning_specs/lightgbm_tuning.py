import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor,CatBoostClassifier
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score
import optuna
from sklearn.model_selection import train_test_split
import pickle

import sys,os
import data.utils.duckdb_utils as ddu
import duckdb
import data.utils.data_utils as du
import random
import lightgbm as lgb
import numpy as np
import sklearn.metrics as mt
from modelling.model_tuning_specs.base import base_model_tuning
import pickle
import pandas as pd
import pickle
import pandas as pd
from collections import Counter
import gc
from optuna.integration import LightGBMPruningCallback


class tuner_model(base_model_tuning):
                 
    def __init__(self,
                 master_config_path, 
                 master_config_name,
                 db_connection,
                 database_path=None, 
                 train_feature_table = None,
                 train_feature_selection_table=None,
                 train_tuning_info_table=None,
                 verbose=True):
        base_model_tuning.__init__(self, 
                                master_config_path=master_config_path, 
                                master_config_name=master_config_name,
                                db_connection=db_connection,
                                database_path=database_path, 
                                train_feature_table = train_feature_table,
                                train_feature_selection_table=train_feature_selection_table,
                                train_tuning_info_table=train_tuning_info_table,
                                verbose=verbose)
    
    
    def get_search_space(self,trial):
        param = {
            "objective": 'multiclass',
            "n_estimators": trial.suggest_int("n_estimators", 200, 5000),
            "metric": "None",
            "boosting": "gbdt",
            #"learning_rate" : trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
            "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
            'num_leaves': trial.suggest_int('num_leaves', 2, 3000, step = 20),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95, step=0.1),
 #           "bagging_fraction": trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            "bagging_freq": trial.suggest_int('bagging_freq', 1, 7),
            "min_child_samples": trial.suggest_int('min_child_samples', 5, 100),
           "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 2000, step=5),
            "max_depth": trial.suggest_int("max_depth", 15, 60),
            "max_bin": trial.suggest_int("max_bin", 200, 300),
            'verbosity': -1
            
        }
        return param 
    

    def define_and_train_model(self,trial,param,train_data,train_data_label,validation_data,validation_data_label):
        def evaluate_macroF1_lgb(truth, predictions):  
            # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
            pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)
            f1 = f1_score(truth, pred_labels, average='macro')
            return ('macroF1', f1, True) 
        
        model = lgb.LGBMClassifier(**param)
            #model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=0, early_stopping_rounds=100,callbacks=[
            #        LightGBMPruningCallback(trial, "multi_logloss")
            #    ])
        model.fit(train_data, train_data_label, 
                  #eval_set=[(validation_data, validation_data_label),(train_data, train_data_label)],
                  eval_set=[(validation_data, validation_data_label)],
                  #eval_names=['valid','train'],
                  verbose=100, early_stopping_rounds=100,eval_metric=evaluate_macroF1_lgb,
                  callbacks=[
                    LightGBMPruningCallback(trial, "macroF1")
                ])
        return model
    
    def evaluate_model(self,model,test_data,test_data_label):
        preds = model.predict(test_data)
        pred_labels = np.rint(preds)
        accuracy = accuracy_score(test_data_label, pred_labels)
        f1 = f1_score(test_data_label, pred_labels,average='macro')

        print(f"ACCURACY VALUE IS {accuracy}")
        print(f"F1 VALUE IS {f1}")
        #metric_val = precision_score(test_data_label, pred_labels,labels=[1,2],average='macro')
        metric_val = precision_score(test_data_label, pred_labels,average='macro')
        print(f"METRIC VALUE IS {metric_val}")
        return f1
            
    def initialize_tuning_type(self):
        self.tuning_type = 'optuna'
        self.algo_name = 'lightgbm'
        
    def define_and_run_study(self):
        study = optuna.create_study(direction='maximize', study_name="lightgbmtune",)
        study.optimize(self.objective_function, n_trials=35) 
        print(study.best_trial)
        print( study.best_trial.user_attrs)
        #best_model = study.best_trial.user_attrs['model']
        best_model=None
        return best_model,dict(study.best_trial.params)
        
