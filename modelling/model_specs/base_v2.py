from config.common.config import Config, DefineConfig
import data.utils.duckdb_utils as ddu
import duckdb
import data.utils.data_utils as du
import pandas as pd
import datetime as dt
import gc
import random
import ast
import numpy as np
from collections import Counter
import pickle
import sys,os
import data.utils.duckdb_utils as ddu
import duckdb
import data.utils.data_utils as du
import random
import lightgbm as lgb
import sklearn.metrics as mt
from collections import Counter
import gc
from pandas.api.types import is_numeric_dtype, is_bool_dtype,   is_datetime64_any_dtype, is_string_dtype, is_datetime64_dtype
from modelling.utils.ml_helper import base_model_helper


class base_model(base_model_helper):
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
        if not ddu.check_if_table_exists(self.db_connection, table_name=self.train_training_info_table):
            create_table_arg = {
                'replace': True, 'table_column_arg': 'label_name VARCHAR,training_algo VARCHAR,train_partition VARCHAR ,test_partition VARCHAR, validation_partition VARCHAR, train_data_shape VARCHAR,test_data_shape VARCHAR,validation_data_shape VARCHAR,feature_names VARCHAR,model_params VARCHAR,metrics_dict_list VARCHAR,model_file_name VARCHAR,label_mapper VARCHAR,model BLOB, feature_table_name VARCHAR,feature_selection_method VARCHAR,feature_importance_df VARCHAR,cat_non_cat_info VARCHAR,updated_timestamp TIMESTAMP'}
            #ddu.create_table(self.db_conection, table_name=self.zip_files_table,
            ddu.create_table(self.db_connection, table_name=self.train_training_info_table, create_table_arg=create_table_arg, df=None)
            du.print_log(f"Table {self.train_training_info_table} created")

    def upsert_feature_training_info_table(self,sql_dict):
        curr_dt = str(dt.datetime.now())
        sql_dict.update({'updated_timestamp':curr_dt})
        if isinstance(sql_dict['train_partition'],list):
            sql_dict.update({'train_partition':"_".join(sorted(sql_dict['train_partition']))})
            sql_dict.update({'validation_partition':"_".join(sorted(sql_dict['validation_partition']))})
            sql_dict.update({'test_partition':"_".join(sorted(sql_dict['test_partition']))})
        update_where_expr = f"validation_partition = '{sql_dict['validation_partition']}' and test_partition = '{sql_dict['test_partition']}' and train_partition = '{sql_dict['train_partition']}' and feature_selection_method ='{sql_dict['feature_selection_method']}' and label_name = '{sql_dict['label_name']}'"
        self.update_insert(sql_dict = sql_dict,
                        table_name = self.train_training_info_table,
                        update_where_expr=update_where_expr)
    
    def model_train(self,model):
        final_info = {}
        #reload_init_model = False
        print(f"Number of features are {len(self.feature_names)}")
        first_iteration = True
        #for col in self.feature_names:
        #    print(f"\t {col}")
        i = 0
        
        for train_combination,validation_combination,test_combination in self.fold_combinations:
            print("#"*100)
            print(f"Training for : {train_combination}")
            print(f"Validation for : {validation_combination}")
            print(f"Testing for : {test_combination}")
            self.get_train_val_data(train_filter=train_combination,
                                    validation_filter=validation_combination,
                                    test_filter = test_combination,
                                    train_data_scamble=self.train_data_scamble,
                                    train_sampling_fraction=self.train_sampling_fraction,
                                    validation_data_scamble=self.validation_data_scamble,
                                    validation_sampling_fraction=self.validation_sampling_fraction)

            self.create_fit_params()

            print(f"Shape of train data is {self.train_data.shape}")
            print(f"Shape of train data label is {self.train_data_label.shape}")
            print(f"Shape of validation data is {self.validation_data.shape}")
            print(f"Shape of validation label data is {self.validation_data_label.shape}")
            print(f"Shape of test data is {self.test_data.shape}")
            print(f"Shape of test data label is {self.test_data_label.shape}")
            #if reload_init_model:
            #    print(f"Initializing weights with previous model")
            #    self.model_fit_params.update({'init_model': model})
            #    reload_init_model = True
            #    print(f"Setting reload_init_model to {reload_init_model}")
            if first_iteration:
                feature_importance_values = np.zeros(len(self.feature_names))
                first_iteration = False
                print(f"Setting first_iteration to {first_iteration}")
            model.fit(**self.model_fit_params)
            metrics_dict_list = self.model_evaluation(model,
                                            test_data=self.test_data,
                                            actual_labels=self.test_data_label,
                                            label_name=self.label_name,
                                            features_name=None,
                                            test_table_name=None,
                                            table_filter=None,
                                            model_path=None,
                                            prob_theshold_list=self.prob_theshold_list)
            final_info.update({'label_name':self.label_name})
            final_info.update({'training_algo':'lightgbm'})
            
            final_info.update({'train_partition':train_combination})
            final_info.update({'test_partition':test_combination})
            final_info.update({'validation_partition':validation_combination})

            #final_info.update({'model_fit_params':model_fit_params})

            final_info.update({'train_data_shape':str(self.train_data.shape).replace(",","-").replace(" ","")})
            final_info.update({'test_data_shape':str(self.test_data.shape).replace(",","-").replace(" ","")})
            final_info.update({'validation_data_shape':str(self.validation_data.shape).replace(",","-").replace(" ","")})
            feature_str="-".join(self.feature_names).replace("'","\\'").replace(",","\\,")
            #model_params_save = str(self.model_params).replace("'","\\'").replace(",","\\,")
            #model_params = "dummy"
            #metrics_dict_list = str(metrics_dict_list).replace("'","\\'").replace(",","\\,").replace("{","\\{").replace("}","\\}").replace("[","\\[").replace("]","\\]")
            #model_param = str(model_param).replace("'","\\'")
            #model_param='dummy'
            tc = train_combination[0].split("_")[-1] if isinstance(train_combination,list) else train_combination.split("_")[-1]
            tst = test_combination[0].split("_")[-1] if isinstance(test_combination,list) else test_combination.split("_")[-1]
            vc = validation_combination[0].split("_")[-1] if isinstance(validation_combination,list) else validation_combination.split("_")[-1]

            #tst = test_combination.split("_")[-1]
            #vc = validation_combination.split("_")[-1]
            metric_file_name = f"{self.train_model_base_path}metric_{self.label_name}_{tc}_{vc}_{tst}.pkl"

            final_info.update({'feature_names':feature_str})
            final_info.update({'model_params':metrics_dict_list})
            final_info.update({'metrics_dict_list':metric_file_name})

            
            model_file_name = f"{self.train_model_base_path}model_{self.label_name}_{tc}_{vc}_{tst}.pkl"
            final_info.update({'model_file_name':model_file_name})
            final_info.update({'label_mapper':self.label_mapper})

            feature_importance_values += model.feature_importances_ 
            
            feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,self.feature_names)), columns=['Value','Feature'])
            print(feature_imp)
            final_info.update({'model':'dummy'})
            final_info.update({'feature_table_name':self.train_feature_table})
            final_info.update({'feature_selection_method':self.feature_selection_method})
            final_info.update({'feature_importance_df':'dummy'})
            final_info.update({'cat_non_cat_info':self.cat_non_cat_info})
            
            self.upsert_feature_training_info_table(final_info)
            print(f" BEST MODEL SCORE FOR THIS ITERATION IS : {model.best_score_}")
    
            print("#"*100)
            print(f" MODEL SAVE AT LOCATION : {model_file_name}")
            self.save_pickle_obj(model_file_name,model)
            self.save_pickle_obj(metric_file_name,metrics_dict_list)
            i = i+1

        feature_importance_values = feature_importance_values/i
        feature_importances = pd.DataFrame({'feature': self.feature_names, 'importance': feature_importance_values})
        fi_file_name = f"{self.train_model_base_path}feature_imp_{self.label_name}.csv"
        print(feature_importances)
        feature_importances.to_csv(fi_file_name)
        return model
    
    def create_fit_params(self):
        print("create_fit_params function is not implemented in base")
    
    def already_ran_columns(self):
        col_list = self.get_prev_trained_data()        
        return col_list
           
    def train_all_labels(self,
                         model_fit_params,
                         prob_theshold_list=None,
                         model_params=None,
                         only_run_for_label = [],
                         forced_labels = [],
                         feature_selection_method='featurewiz',
                         filter_out_cols=None):
        

            
        if prob_theshold_list is not None:
            self.prob_theshold_list = prob_theshold_list
        self.model_params = model_params
        self.model_fit_params = model_fit_params
        
        self.set_attributes(feature_selection_method=feature_selection_method,
                                        only_run_for_label=only_run_for_label,
                                        forced_labels=forced_labels )

        if len(self.feature_selection_dict) > 0:
            for label_name,feature_map_list in self.feature_selection_dict.items():
                print("***********************************************************************")
                print(f"*************** MODEL TRAINING STARTS FOR {label_name} ****************")
                print("***********************************************************************")
                
                self.label_name = label_name
                self.feature_names = feature_map_list[0]
                print(f"Length of feature cols before filtering out {len(self.feature_names)}")
                if filter_out_cols is not None:
                    for cp in filter_out_cols:
                        self.feature_names = [col for col in self.feature_names if (cp not in col)] 
                print(f"Length of actual cols after filtering out {len(self.feature_names)}")
                self.label_mapper=feature_map_list[1]
                if self.model_params is None:
                    self.model_params = self.get_params_trained_data(label_name,self.feature_selection_method,self.algo_name,self.tuning_type)
                print(f"Model parameters are below")
                print(self.model_params)
                #self.fold_combinations=self.get_label_combination(scramble_all=True,comb_diff=3,select_value=4)
                self.fold_combinations = self.get_combination_with_strategy(scramble_all=self.scramble_all,
                                                            comb_diff=self.comb_diff,
                                                            select_value=self.select_value,
                                                            stride=self.stride,
                                                            strategy=self.strategy)
                self.initialize_dtype_info()
                print(f"Number of feature columns before null drop are {len(self.feature_names)}")
                self.feature_names = [col for col in self.feature_names if col not in self.null_cols]
                print(f"Number of feature columns after null drop are {len(self.feature_names)}")
                model = self.create_model()
                print(f"MODEL ARCHITECTURE IS ")
                print(model)
                model = self.model_train(model)
                gc.enable()
                del model
                print(f" MODEL OBJECT THRASHED")
                gc.collect()  
                print("***********************************************************************")
                print(f"************* MODEL TRAINING COMPLETED FOR {label_name} ***************")
                print("***********************************************************************")
        else:
            print(f"feature_selection_dict is empty {self.feature_selection_dict}. No training will start")
                       
    def get_prev_trained_data(self):
        sql = f"select * from {self.train_training_info_table}"
        info = ddu.load_table_df(self.db_connection,table_name=None,column_names=None,filter=None,load_sql=sql)
        col_list = list(info['label_name'].unique())
        return col_list

    def get_params_trained_data(self,label_name,feature_selection_method,algo_name,tuning_type):
        sql = f"select best_tuned_parameters from {self.train_tuning_info_table} where label_name = '{label_name}' and feature_selection_method = '{feature_selection_method}' and algo_name = '{algo_name}' and tuning_type = '{tuning_type}'"
        info = ddu.load_table_df(self.db_connection,table_name=None,column_names=None,filter=None,load_sql=sql)
        best_param = ast.literal_eval(info['best_tuned_parameters'].loc[0])
        print(f"Extracted parameter is {best_param}")
        return best_param
    
    def create_model(self):
        print("Method not implemented in base class")
        model = None
        return model
    
    def model_spec_info(self):
        self.algo_name = 'none'
        self.tuning_type = 'none'