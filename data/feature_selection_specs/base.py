from config.common.config import Config, DefineConfig
import data.utils.duckdb_utils as ddu
import duckdb
import data.utils.data_utils as du
import pandas as pd
import datetime as dt
import gc
import random
from collections import Counter
import numpy as np

class base_feature_selection(DefineConfig):
    def __init__(self, 
                 master_config_path, 
                 master_config_name,
                 db_connection,
                 database_path=None, 
                 train_feature_table = None,
                 train_feature_selection_table=None,
                 train_feature_info_table=None,
                 filter_out_cols = None,
                 ignore_cols = None,
                 verbose=True):
        DefineConfig.__init__(self, master_config_path, master_config_name)
        self.database_path = database_path
        if train_feature_selection_table is not None:
            self.train_feature_selection_table = train_feature_selection_table
            
        if train_feature_info_table is not None:
            self.train_feature_info_table = train_feature_info_table

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
        self.ignore_cols = ignore_cols
        self.table_setup()
        du.print_log(f"Table setup completed")

    def table_setup(self):
        du.print_log(f"Table name is {self.train_feature_selection_table}")
        if not ddu.check_if_table_exists(self.db_connection, table_name=self.train_feature_selection_table):
            create_table_arg = {
                'replace': True, 'table_column_arg': 'label_name VARCHAR,feature_selection_method VARCHAR,feature_name VARCHAR ,selection_flag VARCHAR, feature_importance DOUBLE, label_mapper VARCHAR,updated_timestamp TIMESTAMP'}
            #ddu.create_table(self.db_conection, table_name=self.zip_files_table,
            ddu.create_table(self.db_connection, table_name=self.train_feature_selection_table, create_table_arg=create_table_arg, df=None)
            du.print_log(f"Table {self.train_feature_selection_table} created")

    def upsert_feature_selection_table(self,label_name,feature_selection_method,feature_name,selection_flag,feature_importance,label_mapper):
        curr_dt = str(dt.datetime.now())
        sql_dict = {"label_name":label_name,"feature_selection_method":feature_selection_method,"feature_name":feature_name,
                    "selection_flag":selection_flag,"feature_importance":feature_importance,"label_mapper":label_mapper,
                    "updated_timestamp":curr_dt}
        update_where_expr = f"feature_name = '{feature_name}' and feature_selection_method ='{feature_selection_method}' and label_name = '{label_name}'"
        self.update_insert(sql_dict = sql_dict,
                        table_name = self.train_feature_selection_table,
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
            
    def get_feature_info(self):
        sql = f"select * from {self.train_feature_info_table}"
        self.info = ddu.load_table_df(self.db_connection,table_name=None,column_names=None,filter=None,load_sql=sql)
        sql = f"select distinct time_split from {self.train_feature_table}"
        timesplit_df = ddu.load_table_df(self.db_connection,table_name=None,column_names=None,filter=None,load_sql=sql)
        self.time_splits = [col for col in timesplit_df['time_split'].tolist() if str(col) != 'nan']
        self.train_split = [col for col in self.time_splits if 'train' in col]
        self.labels = self.info[self.info['column_type'] == 'label']['feature_name'].tolist()
        self.nulls_cols = self.info[self.info['column_type'] == 'null_reject']['feature_name'].tolist()
        nulls_cols = [x.replace("count_nulls_",'') for x in self.nulls_cols]
        self.unstable_cols = self.info[self.info['column_type'] == 'unstable_column']['feature_name'].tolist()
        #features = info[(info['column_type'] == 'feature') & (info['feature_name'] != 'timestamp')]['feature_name'].tolist()
        self.features = self.info[(self.info['column_type'] == 'feature')]['feature_name'].tolist()
        self.features = [col for col in self.features if col not in nulls_cols]
        print(f"Number of features after removing nulls cols {len(self.features)}")
        if self.ignore_cols is not None:
            self.features = [col for col in self.features if col not in self.ignore_cols]
            print(f"Number of features after ignoring cols {len(self.features)}")
        self.features = [x.strip() for x in self.features]
        print(f"Number of features are {len(self.features)}")
        self.labels = [x.strip() for x in self.labels]
        print(f"Number of labels are {len(self.labels)}")
        print(f"Number of unstable column are {len(self.unstable_cols)}")
        print(f"Number of labels are {len(self.labels)}")
        print(f"Number of null columns are {len(self.nulls_cols)}")
        #label = labels[0]
    
    def get_info_for_feature_selection(self,
                                        label,
                                        time_split=['time_split_train_fold_10']):
        
        time_split = "','".join(time_split)
        time_split = f"('{time_split}')"

        sql = f"select {','.join(self.features)},{label} from {self.train_feature_table} where time_split in {time_split}"
        print(sql)
        df_features = ddu.load_table_df(self.db_connection,table_name=None,column_names=None,filter=None,load_sql=sql)
        print(f"Shape of features loaded is {df_features.shape}")

        df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        null_cols = du.checknans(df_features, threshold=100)
        print(f"Null columns are {null_cols}")
        print(f"Shape of df_features before null drop {df_features.shape}")
        df_features = df_features.drop(null_cols, axis=1)
        print(f"Shape of df_features after null drop {df_features.shape}")
        df_features = df_features.dropna()
        print(f"Shape of df_features after nan drop {df_features.shape}")
        df_features = df_features[df_features[label] != 'unknown']
        dtypes_df = pd.DataFrame(df_features.dtypes).reset_index()
        dtypes_df.columns = ['col_name','data_type']
        print(dtypes_df['data_type'].value_counts())

        #df_features[label] = df_features[label].map(label_mapper)
        nulldf = du.nullcolumns(df_features)
        print(nulldf)
        df_features = df_features.reset_index(drop=True)
        actual_cols = [col for col in df_features.columns.tolist() if col.split("_")[-1] != '2']
        #col_pattern =["CMF_","RVGIs_","SSF_","STDEVALL","DPO_","ICS_","KAMA_","PDIST","RVGI_","PSAR","QS_"]
        #all_col = df_features.tolist()
        #no_all_cols = []
        print(f"Length of actual cols is {len(actual_cols)}")
        if self.filter_out_cols is not None:
            for cp in self.filter_out_cols:
                actual_cols = [col for col in actual_cols if (cp not in col)] 
            print(f"Length of actual cols after filtering out {len(actual_cols)}")
        df_features = df_features[actual_cols]
        print(f"Final shape of df_features is  {df_features.shape}")
        return df_features
    
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
        self.dfcolumns = tmpdf.columns.tolist()
        self.dfcolumns_nottarget = [
            col for col in self.dfcolumns if col != label_name]
        for col in self.dfcolumns_nottarget:
            col_unique_cnt = tmpdf[col].nunique()
            dtype = self.get_datatype(tmpdf[col])
            if dtype == 'NUMERIC' and col_unique_cnt <=cat_threshold:
                info_dict.update({col:['categorical',dtype,col_unique_cnt]})
            elif dtype == 'NUMERIC' and col_unique_cnt > cat_threshold:
                info_dict.update({col:['noncategorical',dtype,col_unique_cnt]})
            elif dtype == 'OBJECT':
                info_dict.update({col:['categorical',dtype,col_unique_cnt]})
            elif dtype == 'TIMESTAMP':
                info_dict.update({col:['timestamp',dtype,col_unique_cnt]})    
            elif dtype == 'UNKNOWN':
                info_dict.update({col:['unknown',dtype,col_unique_cnt]})   
            else:
                info_dict.update({col:['unknown',dtype,col_unique_cnt]})                         
        return info_dict
    
    def get_label_mapper(self,df_features,label):
        counter_obj = dict(Counter(df_features[label].tolist()))
        t = sorted(counter_obj.items(),key=lambda x:x[1],reverse=True)
        label_mapper = {j[0]:i for i,j in enumerate(t,0)}
        #label_mapper = {}
        #for i,col in enumerate(df_features[label].unique().tolist(),0):
        #    label_mapper.update({col:i})
        #print("label mapper is")
        #print(label_mapper)
        return label_mapper
    
    def perform_feature_selection(self,df_features,label):
        print(f"perform_feature_selection method is not implemented in base. returning empty list")
        selected_features=[]
        return selected_features
    
    def set_feature_selection_name(self):
        self.feature_selection_method = None
        
    def run_feature_selection(self,forced_labels=[],split_selection_limit = 2):
        self.get_feature_info()
        if len(forced_labels) > 0:
            self.labels = forced_labels
        for label in self.labels:
            print("*"*100)
            print(f"****** SELECTING FEATURES for LABEL {label} *******")
            print("*"*100)
            final_split = []
            selecting_split = self.train_split.copy()
            print(f"Total splits are {selecting_split}")
            for i in range(split_selection_limit):
                selected = random.choice(selecting_split)
                print(f"Split selected is {selected}")
                final_split.append(selected)
                selecting_split.remove(selected)
            print(f"Selected splits are {final_split}")
            df_features = self.get_info_for_feature_selection(label=label,time_split=final_split)
            label_mapper = self.get_label_mapper(df_features,label)
            df_features[label] = df_features[label].map(label_mapper)
            final_cols = [col for col in df_features.columns.tolist() if col not in ['close','high','low','open','timestamp']]
            print(f"df_features before filtering ohlc columns  : {df_features.shape}")
            df_features = df_features[final_cols]
            print(f"df_features after filtering ohlc columns  : {df_features.shape}")
            selected_features = self.perform_feature_selection(df_features,label)
            for col,f_imp in selected_features.items():
                self.upsert_feature_selection_table(label_name=label,
                                                    feature_selection_method=self.feature_selection_method,
                                                    feature_name=col,
                                                    selection_flag = 'Y',
                                                    feature_importance=f_imp,
                                                    label_mapper=label_mapper)
            print("*"*100)
            print(f"******* FEATURE SELECTION COMPLETED FOR LABEL {label} ********")
            print("*"*100)
            gc.enable()
            del df_features
            gc.collect()