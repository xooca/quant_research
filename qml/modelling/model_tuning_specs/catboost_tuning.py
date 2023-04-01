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
from qml.modelling.model_tuning_specs.base import base_model_tuning
import pickle
import pandas as pd
import pickle
import pandas as pd
from collections import Counter
import gc
from optuna.integration import CatBoostPruningCallback
from qml.modelling.metrics import metrics as met

class GiniMetricAllNormalized(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        # approxes is list of indexed containers (containers with only __len__ and __getitem__ defined), one container
        # per approx dimension. Each container contains floats.
        # weight is one dimensional indexed container.
        # target is float.

        # weight parameter can be None.
        # Returns pair (error, weights sum)
        print(approxes)
        print('*****')
        print(target)
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]
        error_sum = 0.0
        weight_sum = 1.0
        error_sum = met.calculate_gini_for_all_class(target, approx,normalized=True)
        return error_sum, weight_sum

class GiniMetricMinorityNormalized(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        # approxes is list of indexed containers (containers with only __len__ and __getitem__ defined), one container
        # per approx dimension. Each container contains floats.
        # weight is one dimensional indexed container.
        # target is float.

        # weight parameter can be None.
        # Returns pair (error, weights sum)
        #print(approxes)
        #print('*****')
        #print(target)
        #assert len(approxes) == 1
        #assert len(target) == len(approxes[0])
        approx = approxes[0]
        error_sum = 0.0
        weight_sum = 1.0
        error_sum = met.calculate_gini_for_minority_class(target, approx,normalized=True)
        return error_sum, weight_sum

class GiniMetricAll(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        # approxes is list of indexed containers (containers with only __len__ and __getitem__ defined), one container
        # per approx dimension. Each container contains floats.
        # weight is one dimensional indexed container.
        # target is float.

        # weight parameter can be None.
        # Returns pair (error, weights sum)
        #print(approxes)
        #print('*****')
        #print(target)
        #assert len(approxes) == 1
        #assert len(target) == len(approxes[0])
        approx = approxes[0]
        error_sum = 0.0
        weight_sum = 1.0
        error_sum = met.calculate_gini_for_all_class(target, approx,normalized=False)
        return error_sum, weight_sum

class GiniMetricMinority(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        # approxes is list of indexed containers (containers with only __len__ and __getitem__ defined), one container
        # per approx dimension. Each container contains floats.
        # weight is one dimensional indexed container.
        # target is float.

        # weight parameter can be None.
        # Returns pair (error, weights sum)
        #print(approxes)
        #print('*****')
        #print(target)
        #assert len(approxes) == 1
        #assert len(target) == len(approxes[0])
        approx = approxes[0]
        error_sum = 0.0
        weight_sum = 1.0
        error_sum = met.calculate_gini_for_minority_class(target, approx,normalized=False)
        return error_sum, weight_sum

class ProfitMetricAll:
    @staticmethod
    def get_profit(y_true, y_pred):
        loss = met.calculate_profit_cost_for_all_class(y_true, y_pred,tp_w=3,fn_w=2,fp_w=2,tn_w=0)
        return loss
    
    def is_max_optimal(self):
        return True # greater is better

    def evaluate(self, approxes, target, weight):  
        approx = expit(approxes)
        approx = np.array(approx).argmax(axis=0)
        y_true = np.array(target).astype(int)   
        score = self.get_profit(y_true, approx)
        return score, 1

    def get_final_error(self, error, weight):
        return error

class ProfitMetricMinority:
    @staticmethod
    def get_profit(y_true, y_pred):
        loss = met.calculate_profit_cost_for_minority_class(y_true, y_pred,tp_w=3,fn_w=4,fp_w=4,tn_w=0)
        return loss
    
    def is_max_optimal(self):
        return True # greater is better

    def evaluate(self, approxes, target, weight):   
        approx = expit(approxes)
        approx = np.array(approx).argmax(axis=0)
        y_true = np.array(target).astype(int)   
        score = self.get_profit(y_true, approx)
        return score, 1

    def get_final_error(self, error, weight):
        return error

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
                "objective": trial.suggest_categorical("objective", ["CrossEntropy","Logloss"]),
                #"eval_metric" : trial.suggest_categorical("eval_metric", ["Precision","AUC","BalancedAccuracy","Recall","F","F1"]),
                "bootstrap_type" : trial.suggest_categorical("bootstrap_type", ["Bayesian","Bernoulli"]),
                "learning_rate" : trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 0, 100, step=5),
                #"bagging_temperature": trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
                #"use_best_model": True,
                "depth": trial.suggest_int("depth", 6, 16),
                "min_data_in_leaf" : trial.suggest_int("min_data_in_leaf", 1, 10),
               # "max_leaves" : trial.suggest_int("max_leaves", 20, 40),
                "auto_class_weights":trial.suggest_categorical("auto_class_weights", ["Balanced","SqrtBalanced", None]),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
                "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                "random_strength" :trial.suggest_int('random_strength', 0, 100)                }
        else:
            param = {
                
                "objective": trial.suggest_categorical("objective", ["MultiClass","MultiClassOneVsAll"]),
               # "eval_metric" : trial.suggest_categorical("eval_metric", [ProfitMetric(),"Kappa","WKappa","Accuracy","MCC","AUC","F1"]),
                "bootstrap_type" : trial.suggest_categorical("bootstrap_type", ["Bayesian","Bernoulli"]),
                "learning_rate" : trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 0, 100, step=5),
                #"bagging_temperature": trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
                #"use_best_model": True,
                "depth": trial.suggest_int("depth", 6, 16),
                "min_data_in_leaf" : trial.suggest_int("min_data_in_leaf", 1, 10),
               # "max_leaves" : trial.suggest_int("max_leaves", 20, 40),
                "auto_class_weights":trial.suggest_categorical("auto_class_weights", ["Balanced","SqrtBalanced", None]),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
                "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                "random_strength" :trial.suggest_int('random_strength', 0, 100)
                }
        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)            
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

    def define_and_train_model(self,trial,param):
        if self.convert_to_cat:
            param.update({'cat_features':self.cat_cols})
        model = CatBoostClassifier(**param)
        self.create_fit_params()
        #self.model_fit_params.update({'callbacks':[CatBoostPruningCallback(trial, ProfitMetric(),0)]})
        model.fit(**self.model_fit_params)
        return model
    
    def evaluate_model(self,model):
        preds = model.predict(self.test_data)
        pred_labels = np.rint(preds)
        accuracy = accuracy_score(self.test_data_label, pred_labels)
        f1 = f1_score(self.test_data_label, pred_labels,average='macro')

        print(f"ACCURACY VALUE IS {accuracy}")
        print(f"F1 VALUE IS {f1}")
        #metric_val = precision_score(test_data_label, pred_labels,labels=[1,2],average='macro')
        metric_val = precision_score(self.test_data_label, pred_labels,average='macro')
        print(f"METRIC VALUE IS {metric_val}")
        return f1
            
    def model_spec_info(self):
        self.tuning_type = 'optuna'
        self.algo_name = 'catboost'
        self.add_cat_col_to_model = True
        self.convert_to_cat = True
        self.if_return_df = True
        
        self.train_data_scamble=True
        self.train_sampling_fraction=1
        self.validation_data_scamble=True
        self.validation_sampling_fraction=1
        
        self.scramble_all=True
        self.comb_diff=3
        self.select_value=4
        self.stride=2
        self.strategy='chunking'
        self.feature_strategy = 'selection' # 'all-notinclude-unstable-feature','all-include-unstable-feature','selection'
        self.standardize_strategy = None # 'all','only-unstable'
        self.study_name = "catboost_tune"
        
    def define_and_run_study(self):
        study = optuna.create_study(direction='maximize', study_name=self.study_name)
        study.optimize(self.objective_function, n_trials=35) 
        print(study.best_trial)
        print( study.best_trial.user_attrs)
        #best_model = study.best_trial.user_attrs['model']
        best_model=None
        return best_model,dict(study.best_trial.params)