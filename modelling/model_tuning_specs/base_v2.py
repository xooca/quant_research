from config.common.config import Config, DefineConfig
import data.utils.duckdb_utils as ddu
import duckdb
import data.utils.data_utils as du
import pandas as pd
import datetime as dt
import gc
import random
from collections import Counter
import ast
import numpy as np
import pickle
from modelling.utils.ml_helper import base_model_helper
                                               

class base_model_tuning(base_model_helper):
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
                 verbose=True):
        base_model_helper.__init__(self, master_config_path=master_config_path, 
                                        master_config_name=master_config_name,
                                        db_connection=db_connection,
                                        database_path=database_path, 
                                        train_feature_table = train_feature_table,
                                        train_feature_selection_table=train_feature_selection_table,
                                        train_feature_info_table=train_feature_info_table,
                                        train_training_info_table=train_training_info_table,
                                        train_tuning_info_table=train_tuning_info_table,
                                        verbose=verbose)
        self.table_setup()

    def table_setup(self):
        du.print_log(f"Table name is {self.train_tuning_info_table}")
        if not ddu.check_if_table_exists(self.db_connection, table_name=self.train_tuning_info_table):
            create_table_arg = {
                'replace': True, 'table_column_arg': 'label_name VARCHAR,feature_selection_method VARCHAR,feature_names VARCHAR ,algo_name VARCHAR, tuning_type VARCHAR, best_tuned_parameters VARCHAR, parameter_tune_info VARCHAR,updated_timestamp TIMESTAMP'}
            #ddu.create_table(self.db_conection, table_name=self.zip_files_table,
            ddu.create_table(self.db_connection, table_name=self.train_tuning_info_table, create_table_arg=create_table_arg, df=None)
            du.print_log(f"Table {self.train_tuning_info_table} created")
 
    def upsert_tuning_info_table(self,sql_dict):
        curr_dt = str(dt.datetime.now())
        #sql_dict.update({"updated_timestamp":curr_dt})
        sql_dict['updated_timestamp'] = curr_dt
        update_where_expr = f"label_name = '{sql_dict.get('label_name')}' and feature_selection_method ='{sql_dict.get('feature_selection_method')}' and algo_name ='{sql_dict.get('algo_name')}' and tuning_type ='{sql_dict.get('tuning_type')}' "
        self.update_insert(sql_dict = sql_dict,
                table_name = self.train_tuning_info_table,
                update_where_expr=update_where_expr)
                    
    def create_model(self,model_params):
        print("Method not implemented in base class")
        model = None
        return model
    
    def already_ran_columns(self):
        col_list = self.get_prev_tuned_data()        
        return col_list
               
    def tune_all_labels(self,
                         only_run_for_label = [],
                         forced_labels = [],
                         feature_selection_method='featurewiz',
                         limit=2):
        self.set_attributes(feature_selection_method=feature_selection_method,
                            only_run_for_label=only_run_for_label,
                            forced_labels=forced_labels )
        self.limit = limit
        #if len(force_tuning_labels)> 0:
        #    already_trained_labels = [c for c in already_trained_labels if c not in force_tuning_labels]

        if len(self.feature_selection_dict) > 0:
            for label_name,feature_map_list in self.feature_selection_dict.items():
                print(f"MODEL TRAINING STARTS FOR {label_name}")
                #model = self.create_model(model_params)
                #self.fold_combinations=self.get_label_combination(scramble_all=True,comb_diff=4,select_value=5)
                self.fold_combinations = self.get_combination_with_strategy(scramble_all=self.scramble_all,
                                                            comb_diff=self.comb_diff,
                                                            select_value=self.select_value,
                                                            stride=self.stride,
                                                            strategy=self.strategy)
                self.label_name = label_name
                self.feature_names = feature_map_list[0]
                self.label_mapper=feature_map_list[1]
                self.initialize_dtype_info()
                self.model_tune()              
        else:
            print(f"feature_selection_dict is empty {self.feature_selection_dict}. No training will start")

    def model_spec_info(self):
        self.tuning_type = 'none'
        self.algo_name = 'none'
    
    def model_tune(self):
        print(self.fold_combinations)
        if len(self.fold_combinations) <= self.limit:
            limit_val = len(self.fold_combinations)
        else:
            limit_val = self.limit
        self.selected_folds = random.sample(self.fold_combinations, limit_val)
        
        best_model,best_param = self.define_and_run_study()
        
        #best_param = {i:str(j) for i,j in best_param.items()}
        modelbestparam_file_name = f"{self.train_model_base_path}bestmodelparam_{self.selected_label_name}.pkl"

        return_dict = {"label_name":self.selected_label_name,"feature_selection_method":self.feature_selection_method,
                       "feature_names":"None","algo_name":self.algo_name,"tuning_type":self.tuning_type,
                       "best_tuned_parameters":best_param,"parameter_tune_info":"None"}
        self.upsert_tuning_info_table(return_dict)

        self.save_pickle_obj(modelbestparam_file_name,best_param)

        gc.enable()  
        del self.selected_folds,self.selected_feature_names,self.selected_label_name,self.selected_label_mapper
        gc.collect()
    
    def define_and_run_study(self):
        print(f"define_and_run_study not implemented in base class")
        best_model= None
        best_param = None
        return best_model,best_param
    
    def get_search_space(self,trial):
        print(f"get_search_space not implemented in base class")
        model = None
        return model
    
    def define_and_train_model(self,trial,param):
        print(f"define_and_train_model not implemented in base class")
        model = None
        return model
    
    def evaluate_model(self,model):
        print(f"evaluate_model not implemented in base class")
        metric=None
        return metric
    
    def objective_function(self,trial):
        metric_values = np.empty(len(self.selected_folds))
        i = 0
        for train_combination,validation_combination,test_combination in self.selected_folds:
            self.get_train_val_data(train_filter=train_combination,
                                    validation_filter=validation_combination,
                                    test_filter = test_combination,
                                    train_data_scamble=self.train_data_scamble,
                                    train_sampling_fraction=self.train_sampling_fraction,
                                    validation_data_scamble=self.validation_data_scamble,
                                    validation_sampling_fraction=self.validation_sampling_fraction)
            param = self.get_search_space(trial)
            model = self.define_and_train_model(trial,param)
            metric_val = self.evaluate_model(model)
            metric_values[i] = metric_val
            i = i+1
        gc.enable()
        del self.train_data,self.train_data_label,self.validation_data,self.validation_data_label,self.test_data,self.test_data_label
        gc.collect()
        mean_metric = np.mean(metric_values)
        print(f"FINAL METRIC IS {mean_metric}")
        return mean_metric