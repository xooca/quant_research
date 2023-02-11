import sys,os
import data.utils.duckdb_utils as ddu
import duckdb
import data.utils.data_utils as du
import random
import lightgbm as lgb
import numpy as np
import sklearn.metrics as mt
from modelling.model_specs.base import base_model
import pickle
import pandas as pd
import pickle
import pandas as pd
from collections import Counter
import gc
from sklearn.preprocessing import MinMaxScaler
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


class base_model_helper(DefineConfig):
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
        DefineConfig.__init__(self, master_config_path, master_config_name)
        self.database_path = database_path
        if train_feature_selection_table is not None:
            self.train_feature_selection_table = train_feature_selection_table
            
        if train_feature_info_table is not None:
            self.train_feature_info_table = train_feature_info_table

        if train_feature_table is not None:
            self.train_feature_table = train_feature_table

        if train_training_info_table is not None:
            self.train_training_info_table = train_training_info_table
 
        if train_tuning_info_table is not None:
            self.train_tuning_info_table = train_tuning_info_table                                  
        #self.train_feature_table = train_feature_table
        #self.train_feature_info_table = train_feature_info_table
        self.verbose = verbose
        if self.database_path is None:
            self.db_connection = db_connection
        else:
            self.db_connection = duckdb.connect(database=self.database_path , read_only=False)
                
    def update_insert(self,sql_dict,table_name,update_where_expr):
        sql_dict_updated = {i:j for i,j in sql_dict.items() if j is not None}
        l = [f"{i}={j}" if (isinstance(j,int) or isinstance(j,float) or isinstance(j,dict) or isinstance(j,list) or isinstance(j,pd.DataFrame)  ) else f"{i}='{j}'"for i,j in sql_dict_updated.items()]
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
            #sql_dict_updated = {i: (f'{str(j)}' if j is None else j) for i,j in sql_dict.items() }
            #sql_dict_updated = {i: (j if j is None else j) for i,j in sql_dict.items() }
            sql_dict_updated = {i:j for i,j in sql_dict.items() if j is not None}
            sql_dict_updated = {i:(j if (isinstance(j,int) or isinstance(j,float) or isinstance(j,dict) or isinstance(j,list) or isinstance(j,pd.DataFrame) )  else f'{j}')for i,j in sql_dict_updated.items()}

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
    
    def upsert_data_in_table(self,sql_dict,update_where_expr,table_name):
        curr_dt = str(dt.datetime.now())
        sql_dict.update({"updated_timestamp":curr_dt})
        self.update_insert(sql_dict = sql_dict,
                        table_name = table_name,
                        update_where_expr=update_where_expr)
                              
    def create_fit_params(self):
        print("create_fit_params function is not implemented in base")
    
    def already_ran_columns(self):
        already_available_cols = []
        print("already_ran_columns column not implemented in ml_helper. Please implement in respective specification or base class")
        return already_available_cols
    
    def model_spec_info(self):
        print("model_spec_info is not implemented in ml_helper. Implement in specification or base file")
        
    def set_attributes(self,
                        feature_selection_method,
                        only_run_for_label,
                        forced_labels):
        self.model_spec_info()
        self.feature_selection_method = feature_selection_method
        self.get_feature_selection_info(feature_selection_method=self.feature_selection_method)

        print(self.feature_selection_dict)
        
        if len(only_run_for_label)>0:
            self.feature_selection_dict = {i:j for i,j in self.feature_selection_dict.items() if i in only_run_for_label}
        else:
            already_trained_labels = self.already_ran_columns()
            print(f"Already present labels for {'.'.join(already_trained_labels)}")
            if len(forced_labels)> 0:
                already_trained_labels = [c for c in already_trained_labels if c not in forced_labels]
            print(f"After filtering out forced labels, already trained labels are {'.'.join(already_trained_labels)}")
            self.feature_selection_dict = {i:j for i,j in self.feature_selection_dict.items() if i not in already_trained_labels}

    def get_feature_info_df(self):
        sql = f"select * from {self.train_feature_info_table}"
        info = ddu.load_table_df(self.db_connection,table_name=None,column_names=None,filter=None,load_sql=sql)
        unstable_columns = info[info['column_type']=='unstable_column']['feature_name'].tolist()
        feature_columns = info[info['column_type']=='feature']['feature_name'].tolist()
        return info,unstable_columns,feature_columns

    def get_feature_selection_info_df(self,feature_selection_method):
        sql = f"select * from {self.train_feature_selection_table} where feature_selection_method='{feature_selection_method}' and selection_flag = 'Y'"
        info = ddu.load_table_df(self.db_connection,table_name=None,column_names=None,filter=None,load_sql=sql)
        return info
    
    def standardize_stratified(self,df,columns):
        for col in df:
            if col in columns:
                df[col] = df[col].astype(np.float)
                df[col] = df[col].replace(np.inf, 0.0)
                if df[col].mean() > 1000:
                    scaler = MinMaxScaler(feature_range=(0, 10))
                    df[col] = scaler.fit_transform(
                        np.asarray(df[col]).reshape(-1, 1))
                elif df[col].mean() > 100:
                    scaler = MinMaxScaler(feature_range=(0, 5))
                    # print(col)
                    df[col] = scaler.fit_transform(
                        np.asarray(df[col]).reshape(-1, 1))
                elif df[col].mean() > 10:
                    scaler = MinMaxScaler(feature_range=(0, 2))
                    # print(col)
                    df[col] = scaler.fit_transform(
                        np.asarray(df[col]).reshape(-1, 1))
                else:
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    df[col] = scaler.fit_transform(
                        np.asarray(df[col]).reshape(-1, 1))
        return df
    
    def get_features_strategy(self,feature_strategy,feature_selection_method='featurewiz'):
        info_features,unstable_columns,feature_columns = self.get_feature_info_df()
        if feature_strategy == 'all-include-unstable-feature':
            #info,unstable_columns,feature_columns = self.get_feature_info_df()
            ret_features = unstable_columns + feature_columns
        elif feature_strategy == 'all-notinclude-unstable-feature':
            #info,unstable_columns,ret_features = self.get_feature_info_df()
            ret_features = feature_columns
        else:
            ret_features = None
        info = self.get_feature_selection_info_df(feature_selection_method)
        return info, info_features,ret_features,unstable_columns
    
    def get_feature_selection_info(self,feature_selection_method='featurewiz'):
        self.info, self.info_features, features, self.unstable_columns = self.get_features_strategy(feature_strategy=self.feature_strategy,feature_selection_method=feature_selection_method)
        self.labels = list(set(self.info['label_name'].tolist()))
        self.feature_selection_dict ={}
        for label in self.labels:
            if features is None:
                features = self.info[self.info['label_name']==label]['feature_name'].tolist()
            features = [x.strip() for x in features]
            mapper = self.info[self.info['label_name']==label]['label_mapper'].tolist()
            mapper = mapper[0]
            print(f"Number of features column are {len(features)} for label {label}")
            self.feature_selection_dict.update({label:[features,ast.literal_eval(mapper)]})
            
    def get_feature_selection_info_v2(self,feature_selection_method='featurewiz'):
        sql = f"select * from {self.train_feature_selection_table} where feature_selection_method='{feature_selection_method}' and selection_flag = 'Y'"
        self.info = ddu.load_table_df(self.db_connection,table_name=None,column_names=None,filter=None,load_sql=sql)
        print(self.info )
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
        
    def initialize_dtype_info(self):
        get_index = random.randrange(len(self.fold_combinations))
        filter = self.fold_combinations[get_index][0]
        #sql = f"select {','.join(self.feature_names)} from {self.train_feature_table} where time_split = '{filter}' and {self.label_name} != 'unknown'"
        sql = self.create_sql_for_data_creation(filter=filter)
        print("Check data extract sql is :")
        print(sql)
        check_data = ddu.load_table_df(self.db_connection,table_name=None,column_names=None,filter=None,load_sql=sql)
        check_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.null_cols = du.checknans(check_data, threshold=100)
        print(f"Null columns detected are {self.null_cols}")
        #data = data.drop(null_cols, axis=1)
        self.cat_non_cat_info,self.cat_cols,self.cat_index = self.get_cat_nocat_col(check_data,label_name=self.label_name,cat_threshold=100)
        #for i,j in self.cat_non_cat_info.items():
        #    print(f"column {i} ............ info : {j}")
        print(f"Number of categorical columns are : {len(self.cat_cols)}")
        print(f"Categorical columns are : {self.cat_cols}")
      
    def save_pickle_obj(self,pickle_file_path,pickle_obj):
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(pickle_obj, f)   
    
    def load_pickle_obj(self,pickle_file_path):
        with open(pickle_file_path, 'wb') as f:
            pickle_obj = pickle.load(f)
        return pickle_obj
    
    def load_data_from_sql(self,sql,feature_names,label_name,label_mapper,data_scamble=False,sampling_fraction=1):
        data = ddu.load_table_df(self.db_connection,table_name=None,column_names=None,filter=None,load_sql=sql)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        print(f"Size of data before row null removal is {data.shape}")
        null_cols = du.checknans(data, threshold=100)
        print(f"Null columns are {null_cols}")
        #data = data.drop(null_cols, axis=1)
        data = data.dropna()
        print(f"Size of data after row null removal is {data.shape}")
        if data_scamble:
            data = data.sample(frac=sampling_fraction)
        print("Data label distibution is :")
        print(data[label_name].value_counts())
        data_label = data[[label_name]]
        #feature_names = [col for col in feature_names if col not in null_cols]
        data = data[feature_names]
        data_label[label_name] = data_label[label_name].map(label_mapper)

        return data,data_label
    
    def get_datatype(self, obj):
        if is_numeric_dtype(obj):
            return 'NUMERIC'
        if is_string_dtype(obj):
            return 'OBJECT'
        if is_bool_dtype(obj):
            return 'OBJECT'
        if is_datetime64_any_dtype(obj):
            return 'TIMESTAMP'
        else:
            return 'UNKNOWN'
        
    def get_cat_nocat_col(self, tmpdf, label_name,cat_threshold=100):
        info_dict = {}
        dfcolumns = tmpdf.columns.tolist()
        dfcolumns_nottarget = [col for col in dfcolumns if col != label_name]
        cat_columns = []
        cat_index = []
        for col in dfcolumns_nottarget:
            col_unique_cnt = tmpdf[col].nunique()
            dtype = self.get_datatype(tmpdf[col])
            if dtype == 'NUMERIC' and col_unique_cnt <=cat_threshold:
                dtype_catnoncat = 'categorical'
                cat_columns.append(col)
            elif dtype == 'NUMERIC' and col_unique_cnt > cat_threshold:
                dtype_catnoncat = 'noncategorical'
            elif dtype == 'OBJECT':
                dtype_catnoncat = 'categorical'
                cat_columns.append(col)
            elif dtype == 'TIMESTAMP':
                dtype_catnoncat = 'timestamp'
            elif dtype == 'UNKNOWN':
                dtype_catnoncat = 'unknown'
            else:
                dtype_catnoncat = 'unknown'
            info_dict.update({col:dtype_catnoncat})
            info_dict.update({f"{col}_count":col_unique_cnt})
            info_dict.update({f"{col}_dtype":dtype})
        for col in cat_columns:
            index_no = tmpdf.columns.get_loc(col)
            cat_index.append(index_no)
        return info_dict,cat_columns,cat_index
     
    def convert_col_to_dtype(self,cols,check_first_dtype='float',convert_first_dtype='int',dtype='category'):
        print(self.train_data[cols].dtypes)
        for c in cols:
            print(f"............. {self.train_data[c].dtype}")
            if self.train_data[c].dtype == check_first_dtype:
                self.train_data[c] = self.train_data[c].astype(convert_first_dtype)
                #print(f"########## {self.train_data[c].dtype}")
            if self.test_data[c].dtype == check_first_dtype:
                self.test_data[c] = self.test_data[c].astype(convert_first_dtype)
                #print(f"########## {self.test_data[c].dtype}")
            if self.validation_data[c].dtype == check_first_dtype:
                self.validation_data[c] = self.validation_data[c].astype(convert_first_dtype)
                #print(f"########## {self.validation_data[c].dtype}")
            print(f"Converting column {c} to {dtype} in train, test and validation dataset")
            self.train_data[c] = self.train_data[c].astype(dtype)
            self.test_data[c] = self.test_data[c].astype(dtype)
            self.validation_data[c] = self.validation_data[c].astype(dtype)
    
    def convert_to_arrays(self):
        self.train_data = np.array(self.train_data)
        self.train_data_label = np.array(self.train_data_label)

        self.test_data = np.array(self.test_data)
        self.test_data_label = np.array(self.test_data_label)
        
        self.validation_data = np.array(self.validation_data)
        self.validation_data_label = np.array(self.validation_data_label)        

    def convert_to_standardize(self):
        if self.standardize_strategy == 'all':
            standardize_col = self.feature_names
        elif self.standardize_strategy == 'only-unstable':
            standardize_col = self.unstable_columns
        else:
            standardize_col = self.feature_names
        print(f"Length of standardized column is {len(standardize_col)}")
        if len(standardize_col) > 0:
            self.train_data = self.standardize_stratified(self.train_data,standardize_col)
            self.validation_data = self.standardize_stratified(self.validation_data,standardize_col)
            self.train_data = self.standardize_stratified(self.train_data,standardize_col)
        print("Standardization completed")
        
    def create_sql_for_data_creation(self,filter):
        if isinstance(filter,list):
            sql_filter_string = "','".join(filter)
            sql_filter_string = f"'{sql_filter_string}'"
            sql = f"select {','.join(self.feature_names)},{self.label_name} from {self.train_feature_table} where time_split in ({sql_filter_string}) and {self.label_name} != 'unknown'"
        else:
            sql = f"select {','.join(self.feature_names)},{self.label_name} from {self.train_feature_table} where time_split = '{filter}' and {self.label_name} != 'unknown'"
        return sql
    
    def get_combination_with_strategy(self,scramble_all=False,comb_diff=3,select_value=4,stride=2,strategy='chunking'):
        combinations = self.get_label_combination(scramble_all=scramble_all,comb_diff=comb_diff,select_value=select_value)
        final_combination=[]
        if strategy=='chunking':
            combinations_t = np.array(combinations)
            combinations_t = combinations_t.T
            final_combination = []
            j = 0
            #stride = 2
            chunks = int(combinations_t.shape[1]/stride)
            for i in range(chunks):
                final_combination.append([list(combinations_t[i][j:j+stride]) for i in range(0,3)])
                j = j + stride
            return final_combination
        else:
            return combinations
   
    def get_train_val_data(self,
                            train_filter,
                            validation_filter,
                            test_filter,
                            train_data_scamble=True,
                            train_sampling_fraction=1,
                            validation_data_scamble=True,
                            validation_sampling_fraction=1):
    
        #sql = f"select {','.join(self.feature_names)},{self.label_name} from {self.train_feature_table} where time_split = '{train_filter}' and {self.label_name} != 'unknown'"
        sql = self.create_sql_for_data_creation(train_filter)
        print("Training data extract sql is :")
        print(sql)
        self.train_data,self.train_data_label = self.load_data_from_sql(sql,self.feature_names,
                                                                        self.label_name,
                                                                        self.label_mapper,
                                                                        data_scamble=train_data_scamble,
                                                                        sampling_fraction=train_sampling_fraction)

        #sql = f"select {','.join(self.feature_names)},{self.label_name} from {self.train_feature_table} where time_split = '{validation_filter}'  and {self.label_name} != 'unknown'"
        sql = self.create_sql_for_data_creation(validation_filter)
        print("Validation data extract sql is :")
        print(sql)
        self.validation_data,self.validation_data_label = self.load_data_from_sql(sql,self.feature_names,
                                                                                  self.label_name,
                                                                                  self.label_mapper, 
                                                                                  data_scamble=validation_data_scamble,
                                                                                  sampling_fraction=validation_sampling_fraction)

        #sql = f"select {','.join(self.feature_names)},{self.label_name} from {self.train_feature_table} where time_split = '{test_filter}'  and {self.label_name} != 'unknown'"
        sql = self.create_sql_for_data_creation(test_filter)
        print("Testing data extract sql is :")
        print(sql)
        self.test_data,self.test_data_label = self.load_data_from_sql(sql,self.feature_names,
                                                                      self.label_name,
                                                                      self.label_mapper)
        
        if self.convert_to_cat:
            self.convert_col_to_dtype(cols=self.cat_cols,dtype='category')
        
        if not self.if_return_df:
            self.convert_to_arrays()
        if self.standardize_strategy is not None:
            self.convert_to_standardize()             
        print(f'Training Data Shape: {self.train_data.shape}.... type of object is {type(self.train_data)}' )
        print(f'Validation Data Shape: {self.validation_data.shape}.... type of object is {type(self.validation_data)}' )
        print(f'Testing Data Shape: {self.test_data.shape}.... type of object is {type(self.test_data)}' )


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

    def model_prediction(self,model,test_data,predict_mode='proba',model_path=None):
        if isinstance(test_data,pd.DataFrame):
            #test_data = test_data[[features_name]]
            test_data = np.array(test_data)
        if model is not None:
            if predict_mode == 'proba':
                preds = model.predict_proba(test_data)
            else:
                preds = model.predict(test_data)
        else:
            self.load_pickle_obj(model_path)
            if predict_mode == 'proba':
                preds = model.predict_proba(test_data)
            else:
                preds = model.predict(test_data)     
        return preds    

    def get_metrics(self,actual_labels,preds_predict,preds_proba):
        
        metric_dic={}
        unique_label = np.unique([actual_labels, preds_predict],return_counts=False)
        actual_count = np.unique(actual_labels,return_counts=False)
        pred_count = np.unique(preds_predict,return_counts=False)
        print(Counter(actual_count))
        print(Counter(pred_count))
        
        if len(unique_label)>0:
            cmtx = pd.DataFrame(
                mt.confusion_matrix(actual_labels, preds_predict, labels=unique_label), 
                index=['true:{:}'.format(x) for x in unique_label], 
                columns=['pred:{:}'.format(x) for x in unique_label]
            )
            print("\t","Confusion matrix")
            print(cmtx)
            #metric_dic.update({'confusion_metr':classification_report})
            classification_report = mt.classification_report(actual_labels,preds_predict)
            print("\t","classification_report")
            print(classification_report)
            #metric_dic.update({'classification_report':classification_report})
            balanced_accuracy_score = mt.balanced_accuracy_score(actual_labels,preds_predict)
            print(f"\t balanced_accuracy_score : {balanced_accuracy_score}")
            #print(balanced_accuracy_score)
            metric_dic.update({'balanced_accuracy_score':balanced_accuracy_score})

            accuracy_score = mt.accuracy_score(actual_labels,preds_predict)
            print(f"\t accuracy_score : {accuracy_score}")
            #print(accuracy_score)
            metric_dic.update({'accuracy_score':accuracy_score})
            #top_k_accuracy_score = mt.top_k_accuracy_score(actual_labels,preds_predict)
            #metric_dic.update({'top_k_accuracy_score':top_k_accuracy_score})
            try:
                roc_auc_score = mt.roc_auc_score(actual_labels,preds_proba,multi_class='ovr')
                print(f"\t roc_auc_score : {roc_auc_score}")
                metric_dic.update({'roc_auc_score':roc_auc_score})
            except Exception as e:
                print(f"Exception in calculating roc_auc_score, setting as 0")
                metric_dic.update({'roc_auc_score':0})
            f1_score = mt.f1_score(actual_labels,preds_predict,average='weighted')
            print(f"\t f1_score : {f1_score}")
            metric_dic.update({'f1_score':f1_score})

            #fpr, tpr, thresholds = mt.roc_curve(actual_labels, preds_proba, pos_label=2)
            #auc = mt.auc(fpr,tpr)
            #print("auc")
            #print(auc)
            #metric_dic.update({'auc':auc})
            precision_score = mt.precision_score(actual_labels,preds_predict,average='weighted')
            print(f"\t precision_score : {precision_score}")
            metric_dic.update({'precision_score':precision_score})
            recall_score = mt.recall_score(actual_labels,preds_predict,average='weighted')
            print(f"\t recall_score : {recall_score}")
            metric_dic.update({'recall_score':recall_score})
        else:
            #metric_dic.update({'classification_report':'null'})
            metric_dic.update({'balanced_accuracy_score':0})
            metric_dic.update({'accuracy_score':0})
            metric_dic.update({'roc_auc_score':0})
            metric_dic.update({'f1_score':0})
            metric_dic.update({'precision_score':0})
            metric_dic.update({'recall_score':0})
            print(f"Unique label found is 0")
        return metric_dic

    def model_evaluation(self,
                         model,
                         test_data,
                         actual_labels,
                         label_name=None,
                         features_name=None,
                         test_table_name=None,
                         table_filter=None,
                         model_path=None,
                         prob_theshold_list=[0.8,0.9]):
        #print("MODEL is",model)
        #print("MODEL PATH is",model_path)
        preds_proba = self.model_prediction(model = model,test_data = test_data,predict_mode='proba',model_path=model_path)        
        preds_predict = self.model_prediction(model = model,test_data = test_data,predict_mode='predict',model_path=model_path)
        if actual_labels is None:
            test_data = ddu.load_table_df(connection=self.db_connection,
                                            column_names = features_name,
                                            table_name=test_table_name,
                                            filter=table_filter)
            test_data = np.array(test_data)
            actual_labels = ddu.load_table_df(connection=self.db_connection,
                                            column_names = label_name,
                                            table_name=test_table_name,
                                            filter=table_filter)
            actual_labels = np.array(actual_labels)
        #print(actual_labels.shape)
        actual_labels = actual_labels.squeeze()
        preds_proba_max = preds_proba.max(axis=1)

        #print('-----------')
        #print(preds_predict.shape)
        unique, counts = np.unique(preds_predict, return_counts=True)
        print(f"\t Count of Predicted labels are : {dict(zip(unique, counts))}")
        unique, counts = np.unique(actual_labels, return_counts=True)
        print(f"\t Count of Actual labels are : {dict(zip(unique, counts))}")
        
        #print(preds_predict)
        #print(preds_proba)
        print(f"\t Actual labels shape {actual_labels.shape}")
        print(f"\t preds_proba shape {preds_proba.shape}")
        print(f"\t preds_proba_max shape {preds_proba_max.shape}")
        

        metrics_dict_list = []
        for prob_theshold in prob_theshold_list:
            th_preds_predict=[]
            th_actual_labels = []
            th_preds_proba = []
            for x,y,z,i in zip(actual_labels,preds_predict,preds_proba,preds_proba_max):
                if i >= prob_theshold:
                    th_actual_labels.append(x)
                    th_preds_predict.append(y)
                    th_preds_proba.append(z)
            print(f"*************Test validation using threshold {prob_theshold}*****************")
            metrics_dict2 = self.get_metrics(th_actual_labels,th_preds_predict,th_preds_proba)
            metrics_dict2.update({"threshold_probability":prob_theshold})
            metrics_dict_list.append(metrics_dict2)
        print("*****************Test validation using defaults*****************")
        metrics_dict1 = self.get_metrics(actual_labels,preds_predict,preds_proba)
        metrics_dict1.update({"threshold_probability":'default'})
        metrics_dict_list.append(metrics_dict1)
        #print(f"*************Test validation using threshold {prob_theshold}*****************")
        #metrics_dict2 = get_metrics(th_actual_labels,th_preds_predict,th_preds_proba)

        return metrics_dict_list

