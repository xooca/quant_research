import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor,CatBoostClassifier
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score
import optuna
from sklearn.model_selection import train_test_split
import pickle
from scipy.special import expit
import datetime as dt

import sys,os
import data.utils.duckdb_utils as ddu
import duckdb
import data.utils.data_utils as du
import random
import lightgbm as lgb
import numpy as np
import sklearn.metrics as mt
from modelling.model_tuning_specs.base_v2 import base_model_tuning
from modelling.model_tuning_specs import catboost_tuning as ct

import pickle
import pandas as pd
import pickle
import pandas as pd
from collections import Counter
import gc
from optuna.integration import CatBoostPruningCallback
from modelling.metrics import metrics as met

class model(ct.model):
                 
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
                "use_best_model": True,
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
                #"eval_metric" : trial.suggest_categorical("eval_metric", [ct.ProfitMetricAll(),ct.ProfitMetricMinority(),ct.GiniMetricAll(),ct.GiniMetricMinority()]),
                #"eval_metric" : trial.suggest_categorical("eval_metric", [ct.ProfitMetricMinority()]),
                "eval_metric" : ct.ProfitMetricMinority(),
                #"bootstrap_type" : trial.suggest_categorical("bootstrap_type", ["Bayesian","Bernoulli"]),
                "learning_rate" : trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 0, 100, step=5),
                "bagging_temperature": trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
                "use_best_model": True,
                #"task_type":"GPU",
                "depth": trial.suggest_int("depth", 6, 16),
                "min_data_in_leaf" : trial.suggest_int("min_data_in_leaf", 1, 10),
               # "max_leaves" : trial.suggest_int("max_leaves", 20, 40),
                "auto_class_weights":trial.suggest_categorical("auto_class_weights", ["Balanced","SqrtBalanced", None]),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
                "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                "random_strength" :trial.suggest_int('random_strength', 0, 100)
                }
       # if param["bootstrap_type"] == "Bayesian":
        #    param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        #elif param["bootstrap_type"] == "Bernoulli":
        #    param["subsample"] = trial.suggest_float("subsample", 0.1, 1)            
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
        print(f"TUNE PARAM IS {param}")
        model = CatBoostClassifier(**param)
        self.create_fit_params()
        callback_func = str(param['eval_metric']).split('object')[0].strip().split(".")[-1]
        self.model_fit_params.update({'callbacks':[CatBoostPruningCallback(trial,callback_func)]})
        model.fit(**self.model_fit_params)

        feature_imp= pd.DataFrame({'feature_importance': model.get_feature_importance(), 
                                   'feature_names': self.feature_names}).sort_values(by=['feature_importance'], 
                                                                                      ascending=False)
        print("FEATURE IMPORTANCE IS")
        print(feature_imp)
        curr_dt = str(dt.datetime.now()).replace(".","").replace("-","")
        f_imp_path = f"{self.train_model_base_path}feature_imp_{curr_dt}.csv"
        feature_imp.to_csv(f_imp_path)
        model_path = f"{self.train_model_base_path}tuned_model_{curr_dt}.pkl"
        du.save_object(model_path,model)
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
        