import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor,CatBoostClassifier
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score
import optuna
from sklearn.model_selection import train_test_split
import pickle
from scipy.special import expit

import sys,os
import qml.data.utils.duckdb_utils as ddu
import duckdb
import qml.data.utils.data_utils as du
import random
import lightgbm as lgb
import numpy as np
import sklearn.metrics as mt
import xgboost as xgb

from qml.modelling.model_tuning_specs.base import base_model_tuning
import pickle
import pandas as pd
import pickle
import pandas as pd
from collections import Counter
import gc
from optuna.integration import XGBoostPruningCallback
from qml.modelling.metrics import metrics as met
from xgboost.sklearn import XGBClassifier


def evaluate_macroF1_lgb(truth, predictions):  
    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
    pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)
    f1 = f1_score(truth, pred_labels, average='macro')
    return ('macroF1', f1, True) 

def calculate_xg_profit_cost_for_minority_class(truth, predictions):
    metric = met.calculate_profit_cost_for_minority_class(truth, predictions)
    return 'minority_profit', metric

def calculate_xg_profit_cost_for_all_class(truth, predictions):
    metric = met.calculate_profit_cost_for_all_class(truth, predictions)
    return 'majority_profit', metric

class model(base_model_tuning):
                 
    def __init__(self,
                 master_config_path, 
                 master_config_name,
                 db_connection,
                 database_path=None, 
                 train_feature_table = None,
                 train_feature_selection_table=None,
                 train_feature_info_table = None,
                 train_training_info_table = None,
                 train_tuning_info_table=None,
                 verbose=True):
        base_model_tuning.__init__(self, 
                                master_config_path=master_config_path, 
                                master_config_name=master_config_name,
                                db_connection=db_connection,
                                database_path=database_path, 
                                train_feature_table = train_feature_table,
                                train_feature_selection_table=train_feature_selection_table,
                                train_feature_info_table = train_feature_info_table,
                                train_training_info_table = train_training_info_table,
                                train_tuning_info_table=train_tuning_info_table,
                                verbose=verbose)
    

    def get_search_space(self,trial):
        if len(self.label_mapper) == 2:
            param = {
                "objective": trial.suggest_categorical("objective", ["binary:logistic","binary:logitraw"]),
                "num_class": len(self.label_mapper),
               # "eval_metric" : trial.suggest_categorical("eval_metric", [ProfitMetric(),"Kappa","WKappa","Accuracy","MCC","AUC","F1"]),
                "min_split_loss": trial.suggest_uniform("min_split_loss", 0, 1),
                "booster": "gbtree",
                "learning_rate" : trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                "reg_lambda": trial.suggest_int("reg_lambda", 0, 100, step=5),
                "reg_alpha": trial.suggest_int("reg_alpha", 0, 100, step=5),
                "tree_method": "auto",
                "max_depth": trial.suggest_int("max_depth", 6, 16),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_uniform("subsample", 0.1, 1)}
        else:
            param = {
                
                "objective": trial.suggest_categorical("objective", ["multi:softmax","multi:softprob"]),
                "num_class": len(self.label_mapper),
                
                "eval_metric" : trial.suggest_categorical("eval_metric", [calculate_xg_profit_cost_for_minority_class,calculate_xg_profit_cost_for_all_class]),
                "min_split_loss": trial.suggest_uniform("min_split_loss", 0, 1),
                "booster": "gbtree",
                "learning_rate" : trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                "reg_lambda": trial.suggest_int("reg_lambda", 0, 100, step=5),
                "reg_alpha": trial.suggest_int("reg_alpha", 0, 100, step=5),
                "tree_method": "auto",
                "max_depth": trial.suggest_int("max_depth", 6, 16),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_uniform("subsample", 0.1, 1),
                "disable_default_eval_metric" : False
                }        
        return param 
    

    def create_fit_params(self):
                
        self.model_fit_params = {}
        self.model_fit_params.update({'X':self.train_data})
        self.model_fit_params.update({'y':self.train_data_label})
        
        #self.model_fit_params.update({'eval_set':[(self.validation_data, self.validation_data_label), 
        #                                        (self.train_data, self.train_data_label)]})
        
        self.model_fit_params.update({'eval_set':[(self.validation_data, self.validation_data_label)]})
        #self.model_fit_params.update({'categorical_feature':self.cat_cols})            
        print(f"Elements is model_fit_params are {self.model_fit_params.keys()}")

    def convert_dataset_to_xgboost(self):
        self.train_data = xgb.DMatrix(self.train_data[self.feature_names], label=self.train_data[self.label_name])
        #self.train_label = self.train_data[self.label_name]
        self.test_data = xgb.DMatrix(self.test_data[self.feature_names], label=self.test_data[self.label_name])
        #self.test_label = self.test_data[self.label_name]
        self.validation_data = xgb.DMatrix(self.validation_data[self.feature_names], label=self.validation_data[self.label_name])
        #self.validation_label = self.validation_data[self.label_name]
     
        
    def define_and_train_model(self,trial,param):
        if self.convert_to_cat:
            param.update({'cat_features':self.cat_cols})
        self.create_fit_params()
        #self.convert_dataset_to_xgboost()
        if param['eval_metric'] == calculate_xg_profit_cost_for_all_class:
            callback_func = 'majority_profit'
        else:
            callback_func = 'minority_profit'
        self.model_fit_params.update({'callbacks':[XGBoostPruningCallback(trial,callback_func)]})
        #model = xgb.train(self.model_fit_params, self.train_data, num_boost_round=100)
        model = XGBClassifier(**param)
        model.fit(**self.model_fit_params)
        return model
    
    def evaluate_model(self,model):
        preds = model.predict(self.test_data)
        preds = np.rint(preds)
        preds = preds.squeeze() 
        pred_labels = np.array(self.test_data_label)
        pred_labels = pred_labels.squeeze()
        print("PREDLABEL")
        print(pred_labels)
        print("PREDS")
        print(preds)
        #accuracy = accuracy_score(self.test_data_label, pred_labels)
        #f1 = f1_score(self.test_data_label, pred_labels,average='macro')
        profit_cost_min = met.calculate_profit_cost_for_minority_class(pred_labels,preds)
        profit_cost_maj = met.calculate_profit_cost_for_all_class(pred_labels,preds)
        f1_min,acc_min,p_min,r_min = met.calculate_metrices_all(pred_labels,preds,metric_type='minority')
        f1_maj,acc_maj,p_maj,r_maj = met.calculate_metrices_all(pred_labels,preds,metric_type='majority')

        print(f"profit_cost_min is {profit_cost_min}")
        print(f"profit_cost_maj is {profit_cost_maj}")
        print(f"acc_min is {acc_min}")
        print(f"acc_maj is {acc_maj}")
        print(f"f1_min {f1_min}")
        print(f"f1_maj {f1_maj}")
        print(f"precision_min {p_min}")
        print(f"precision_maj {p_maj}")
        print(f"recall_min {r_min}")
        print(f"recall_maj {r_maj}")
        return profit_cost_min
            
    def model_spec_info(self):
        self.tuning_type = 'optuna'
        self.algo_name = 'catboost_v2'
        self.add_cat_col_to_model = True
        self.convert_to_cat = True
        self.if_return_df = True
        
        self.train_data_scamble=True
        self.train_sampling_fraction=0.6
        self.validation_data_scamble=True
        self.validation_sampling_fraction=0.6
        
        self.reduce_data_size = False
        self.scramble_all=True
        self.comb_diff=3
        self.select_value=4
        self.stride=2
        self.strategy='chunking'
        self.standardize_strategy = None # 'all','only-unstable',None
        self.feature_strategy = 'selection'
        #self.feature_strategy = 'all-include-unstable-feature' # 'all-notinclude-unstable-feature','all-include-unstable-feature','selection'
        #self.standardize_strategy = 'only-unstable' # 'all','only-unstable',None
        self.study_name = "catboost_tune"
        
    def define_and_run_study(self):
        study = optuna.create_study(direction='maximize', study_name=self.study_name)
        study.optimize(self.objective_function, n_trials=35) 
        print(study.best_trial)
        print( study.best_trial.user_attrs)
        #best_model = study.best_trial.user_attrs['model']
        best_model=None
        return best_model,dict(study.best_trial.params)