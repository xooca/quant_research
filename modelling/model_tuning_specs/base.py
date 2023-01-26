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

class base_model_tuning(DefineConfig):
    def __init__(self, 
                 master_config_path, 
                 master_config_name,
                 db_connection,
                 database_path=None, 
                 train_feature_table = None,
                 train_feature_selection_table=None,
                 train_tuning_info_table=None,
                 filter_out_cols = None,
                 verbose=True):
        DefineConfig.__init__(self, master_config_path, master_config_name)
        self.database_path = database_path
        if train_feature_selection_table is not None:
            self.train_feature_selection_table = train_feature_selection_table
            
        if train_tuning_info_table is not None:
            self.train_tuning_info_table = train_tuning_info_table

        if train_feature_table is not None:
            self.train_feature_table = train_feature_table
                       
        #self.train_feature_table = train_feature_table
        #self.train_feature_info_table = train_feature_info_table
        self.filter_out_cols = filter_out_cols
        self.verbose = verbose
        if self.database_path is None:
            self.db_connection = db_connection
        else:
            self.db_connection = duckdb.connect(database=self.database_path , read_only=False)
        self.table_setup()
        du.print_log(f"Table setup completed")

    def table_setup(self):
        du.print_log(f"Table name is {self.train_tuning_info_table}")
        if not ddu.check_if_table_exists(self.db_connection, table_name=self.train_tuning_info_table):
            create_table_arg = {
                'replace': True, 'table_column_arg': 'label_name VARCHAR,feature_selection_method VARCHAR,feature_names VARCHAR ,algo_name VARCHAR, tuning_type VARCHAR, best_tuned_parameters DOUBLE, parameter_tune_info VARCHAR,updated_timestamp TIMESTAMP'}
            #ddu.create_table(self.db_conection, table_name=self.zip_files_table,
            ddu.create_table(self.db_connection, table_name=self.train_tuning_info_table, create_table_arg=create_table_arg, df=None)
            du.print_log(f"Table {self.train_tuning_info_table} created")

    def upsert_data_in_table(self,sql_dict,update_where_expr,table_name):
        curr_dt = str(dt.datetime.now())
        #sql_dict = {"label_name":label_name,"feature_selection_method":feature_selection_method,
        #            "feature_names":feature_names,"algo_name":algo_name,"tuning_type":tuning_type,
        #            "best_tuned_parameters":best_tuned_parameters,"label_mapper":label_mapper,
        #            "updated_timestamp":curr_dt}
        sql_dict.update({"updated_timestamp":curr_dt})
        self.update_insert(sql_dict = sql_dict,
                        table_name = table_name,
                        update_where_expr=update_where_expr)
        
    def upsert_tuning_info_table(self,sql_dict):
        update_where_expr = f"label_name = '{sql_dict.get('label_name')}' and feature_selection_method ='{sql_dict.get('feature_selection_method')}' and algo_name ='{sql_dict.get('algo_name')}' and tuning_type ='{sql_dict.get('tuning_type')}' "
        self.update_insert(sql_dict = sql_dict,
                table_name = self.train_tuning_info_table,
                update_where_expr=update_where_expr)
                
    def update_insert(self,sql_dict,table_name,update_where_expr):
        sql_dict_updated = {i:j for i,j in sql_dict.items() if j is not None}
        l = [f"{i}={j}" if (isinstance(j,int) or isinstance(j,float) or isinstance(j,dict) ) else f"{i}='{j}'"for i,j in sql_dict_updated.items()]
        l = ",".join(l)
        sql = f'''
        UPDATE {table_name}
        SET 
        {l}              
        WHERE {update_where_expr}
        '''
        du.print_log(f"Update sql is {sql}")
        t = self.db_connection.execute(sql)
        update_cnt = t.fetchall()
        update_cnt = update_cnt[0][0] if len(update_cnt) > 0 else 0
        du.print_log(f"Update count is {update_cnt}")
        if update_cnt == 0:
            sql_dict_updated = {i: ( str(j) if j is None else j) for i,j in sql_dict.items() }
            insert_col_expr = [f"tab.{i}" for i,j in sql_dict_updated.items()]
            insert_col_expr = ",".join(insert_col_expr)
            sql = f'''
            insert into {table_name}
            select {insert_col_expr} from
            (select {sql_dict_updated} as tab)
            '''
            du.print_log(f"Insert sql is {sql}")
            a = self.db_connection.execute(sql)
            insert_cnt = a.fetchall()
            insert_cnt = insert_cnt[0][0] if len(insert_cnt) > 0 else 0
            du.print_log(f"Insert count is {insert_cnt}")
    
    def save_pickle_obj(self,pickle_file_path,pickle_obj):
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(pickle_obj, f)   
    
    def load_pickle_obj(self,pickle_file_path):
        with open(pickle_file_path, 'wb') as f:
            pickle_obj = pickle.load(f)
        return pickle_obj
    
    def get_feature_selection_info(self,feature_selection_method='featurewiz'):
        sql = f"select * from {self.train_feature_selection_table} where feature_selection_method='{feature_selection_method}' and selection_flag = 'Y'"
        self.info = ddu.load_table_df(self.db_connection,table_name=None,column_names=None,filter=None,load_sql=sql)
        self.labels = list(set(self.info['label_name'].tolist()))
        self.feature_selection_dict ={}
        for label in self.labels:
            features = self.info[self.info['label_name']==label]['feature_name'].tolist()
            features = [x.strip() for x in features]
            mapper = self.info[self.info['label_name']==label]['label_mapper'].tolist()
            mapper = mapper[0]
            print(f"Number of features column are {len(features)} for label {label}")
            self.feature_selection_dict.update({label:[features,ast.literal_eval(mapper)]})
    
    def get_prev_tuned_data(self):
        sql = f"select * from {self.train_tuning_info_table}"
        info = ddu.load_table_df(self.db_connection,table_name=None,column_names=None,filter=None,load_sql=sql)
        col_list = list(info['label_name'].unique())
        return col_list
    
    def create_model(self,model_params):
        print("Method not implemented in base class")
        model = None
        return model
    
    def shuffle_split_combinations(self,train_split,validation_split,test_split,comb_diff=3,select_value=4):
        random.seed=10
        a = sorted(train_split,key=lambda x: int(x.split("_")[-1])+random.randint(454, 989),reverse=True)

        random.seed=98
        b = sorted(validation_split,key=lambda x: int(x.split("_")[-1])+random.randint(300,555),reverse=True)

        random.seed=45
        c = sorted(test_split,key=lambda x: int(x.split("_")[-1])+random.randint(100,324),reverse=True)

        comb = []
        for i in a:
            s = int(i.split("_")[-1])
            for j in b:
                r = int(j.split("_")[-1])
                for k in c:
                    t = int(k.split("_")[-1])
                    if abs(s - r) > comb_diff and abs(r - t) > comb_diff and abs(t - s) > comb_diff:
                        comb.append([i,j,k])
        random.shuffle(comb)
        print(f"Length of combinations before shuffling {len(comb)}")

        res_comb = []
        train_test =[]
        for c in comb:
            v = random.randint(1,10)
            if v < select_value and ([c[0],c[1]] not in train_test):
                res_comb.append(c)
                train_test.append([c[0],c[1]])
        print(f"Length of combinations before shuffling {len(res_comb)}")
        return res_comb
    
    def get_label_combination(self,scramble_all=False,comb_diff=3,select_value=4):
        sql = f"select distinct time_split from {self.train_feature_table}"
        time_split_df = ddu.load_table_df(self.db_connection,table_name=None,column_names=None,filter=None,load_sql=sql)
        
        time_splits = [col for col in time_split_df['time_split'].tolist() if str(col) != 'nan']
        train_split = [col for col in time_splits if 'train' in col]
        validation_split = [col for col in time_splits if 'validation' in col]
        test_split = [col for col in time_splits if 'test' in col]
        buffer1_split = [col for col in time_splits if 'buffer1' in col]
        buffer2_split = [col for col in time_splits if 'buffer2' in col]

        train_split = sorted(train_split,key=lambda x: int(x.split("_")[-1]),reverse=True)
        validation_split = sorted(validation_split,key=lambda x: int(x.split("_")[-1]),reverse=False)
        test_split = sorted(test_split,key=lambda x: int(x.split("_")[-1]),reverse=True)
        if scramble_all is True:
            train_split = list(set(train_split))
        validation_split = list(set(validation_split))
        test_split = list(set(test_split))
        comb_list = self.shuffle_split_combinations(train_split=train_split,
                                            validation_split=validation_split,
                                            test_split=test_split,
                                            comb_diff=comb_diff,
                                            select_value=select_value)
        return comb_list
        
    def tune_all_labels(self,label_name = None,feature_selection_method='featurewiz',force_tuning_labels=[],limit=2):
        self.get_feature_selection_info(feature_selection_method=feature_selection_method)
        self.feature_selection_method = feature_selection_method
        if label_name is not None:
            self.feature_selection_dict = {i:j for i,j in self.feature_selection_dict.items() if i==label_name}
        already_trained_labels = self.get_prev_tuned_data()
        
        if len(force_tuning_labels)> 0:
            already_trained_labels = [c for c in already_trained_labels if c not in force_tuning_labels]
        print(f"Already present labels for {'.'.join(already_trained_labels)}")
        self.feature_selection_dict = {i:j for i,j in self.feature_selection_dict.items() if i not in already_trained_labels}

        if len(self.feature_selection_dict) > 0:
            for label_name,feature_map_list in self.feature_selection_dict.items():
                print(f"MODEL TRAINING STARTS FOR {label_name}")
                #model = self.create_model(model_params)
                fold_combinations=self.get_label_combination(scramble_all=True,comb_diff=4,select_value=5)
                
                model_tune_dict = {'db_connection':self.db_connection,
                                            'fold_combinations':fold_combinations,
                                            'feature_names':feature_map_list[0],
                                            'feature_table_name':self.train_feature_table,
                                            'label_name':label_name,
                                            'label_mapper':feature_map_list[1],
                                            'limit':limit
                                            } 
                self.model_tune(**model_tune_dict)
                                            
        else:
            print(f"feature_selection_dict is empty {self.feature_selection_dict}. No training will start")
            
    def get_train_val_data(self,feature_names,label_name,label_mapper,
                            train_filter,validation_filter,test_filter,
                            if_return_df = False):
    
        sql = f"select {','.join(feature_names)},{label_name} from {self.train_feature_table} where time_split = '{train_filter}' and {label_name} != 'unknown'"
        print("Training data extract sql is :")
        print(sql)
        train_data = ddu.load_table_df(self.db_connection,table_name=None,column_names=None,filter=None,load_sql=sql)
        train_data = train_data.dropna()
        print("Train Data label distibution is :")
        print(train_data[label_name].value_counts())
        train_data_label = train_data[[label_name]]
        train_data = train_data[feature_names]
        
        train_data_label[label_name] = train_data_label[label_name].map(label_mapper)

        sql = f"select {','.join(feature_names)},{label_name} from {self.train_feature_table} where time_split = '{validation_filter}'  and {label_name} != 'unknown'"
        print("Validation data extract sql is :")
        print(sql)
        validation_data = ddu.load_table_df(self.db_connection,table_name=None,column_names=None,filter=None,load_sql=sql)
        validation_data = validation_data.dropna()
        print("Validation Data label distibution is :")
        print(validation_data[label_name].value_counts())
        validation_data_label = validation_data[[label_name]]
        validation_data = validation_data[feature_names]
        
        validation_data_label[label_name] = validation_data_label[label_name].map(label_mapper)


        sql = f"select {','.join(feature_names)},{label_name} from {self.train_feature_table} where time_split = '{test_filter}'  and {label_name} != 'unknown'"
        print("Testing data extract sql is :")
        print(sql)
        test_data = ddu.load_table_df(self.db_connection,table_name=None,column_names=None,filter=None,load_sql=sql)
        test_data = test_data.dropna()

        print("Test Data label distibution is :")
        print(test_data[label_name].value_counts())
        test_data_label = test_data[[label_name]]
        test_data = test_data[feature_names]
        
        test_data_label[label_name] = test_data_label[label_name].map(label_mapper)
        
        print('Training Data Shape: ', train_data.shape)
        print('Validation Data Shape: ', validation_data.shape)
        print('Testing Data Shape: ', test_data.shape)
        if if_return_df is False:
            train_data = np.array(train_data)
            train_data_label = np.array(train_data_label)
            validation_data = np.array(validation_data)
            validation_data_label = np.array(validation_data_label)
            test_data = np.array(test_data)
            test_data_label = np.array(test_data_label)
        return train_data,train_data_label,validation_data,validation_data_label,test_data,test_data_label

    def initialize_tuning_type(self):
        self.tuning_type = 'none'
        self.algo_name = 'none'
        
    def model_tune(self,model_tune_dict):
        self.selected_folds = random.sample(set(model_tune_dict['fold_combinations']), model_tune_dict['limit'])  
        self.selected_feature_names = model_tune_dict['feature_names']
        self.selected_label_name= model_tune_dict['label_name']
        self.selected_label_mapper = model_tune_dict['label_mapper']
        
        best_model,best_param = self.define_and_run_study()
        self.initialize_tuning_type()
        
        return_dict = {"label_name":self.selected_label_name,"feature_selection_method":self.feature_selection_method,
                       "feature_names":self.selected_feature_names,"algo_name":self.algo_name,"tuning_type":self.tuning_type,
                       "best_tuned_parameters":best_param,"parameter_tune_info":model_tune_dict}
        self.upsert_tuning_info_table(return_dict)
        model_file_name = f"{self.train_model_base_path}bestmodel_{self.selected_label_name}.pkl"
        self.save_pickle_obj(model_file_name,best_model)

        gc.enable()  
        del self.selected_folds,self.selected_feature_names,self.selected_label_name,self.selected_label_mapper,best_model
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
    
    def define_and_train_model(self,trial,param,train_data,train_data_label,validation_data,validation_data_label):
        print(f"define_and_train_model not implemented in base class")
        model = None
        return model
    
        
    def evaluate_model(self,model,test_data,test_data_label):
        print(f"evaluate_model not implemented in base class")
        metric=None
        return metric
    
    def objective_function(self,trial):
        metric_values = np.empty(len(self.selected_folds))
        i = 0
        for train_combination,validation_combination,test_combination in self.selected_folds:
            train_data,train_data_label,validation_data,validation_data_label,test_data,test_data_label = self.get_train_val_data(
                                                                                            feature_names=self.selected_feature_names,
                                                                                            label_name=self.selected_label_name,
                                                                                            label_mapper=self.selected_label_mapper,
                                                                                            train_filter=train_combination,
                                                                                            validation_filter=validation_combination,
                                                                                            test_filter = test_combination,
                                                                                            if_return_df=True)
            
            param = self.get_search_space(trial)    
            model = self.define_and_train_model(trial,param,train_data,train_data_label,validation_data,validation_data_label)
            metric_val = self.evaluate_model(model,test_data,test_data_label)
            metric_values[i] = metric_val
            i = i+1
        gc.enable()
        del train_data,train_data_label,validation_data,validation_data_label,test_data,test_data_label
        gc.collect()
        mean_metric = np.mean(metric_values)
        return mean_metric