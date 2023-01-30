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
from modelling.model_specs.base import base_model
from collections import Counter
import gc



class base_model(DefineConfig):
    def __init__(self, 
                 master_config_path, 
                 master_config_name,
                 db_connection,
                 database_path=None, 
                 train_feature_table = None,
                 train_feature_selection_table=None,
                 train_feature_info_table=None,
                 train_training_info_table=None,
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
                                   
        #self.train_feature_table = train_feature_table
        #self.train_feature_info_table = train_feature_info_table
        self.verbose = verbose
        if self.database_path is None:
            self.db_connection = db_connection
        else:
            self.db_connection = duckdb.connect(database=self.database_path , read_only=False)
        self.table_setup()

    def table_setup(self):
        if not ddu.check_if_table_exists(self.db_connection, table_name=self.train_training_info_table):
            create_table_arg = {
                'replace': True, 'table_column_arg': 
                    'label_name VARCHAR,training_algo VARCHAR,train_partition VARCHAR ,test_partition VARCHAR, validation_partition VARCHAR, train_data_shape VARCHAR,test_data_shape VARCHAR,validation_data_shape VARCHAR,feature_names VARCHAR,model_params VARCHAR,metrics_dict_list VARCHAR,model_file_name VARCHAR,label_mapper VARCHAR,model BLOB, feature_table_name VARCHAR,feature_selection_method VARCHAR,feature_importance_df VARCHAR,updated_timestamp TIMESTAMP'}
            #ddu.create_table(self.db_conection, table_name=self.zip_files_table,
            ddu.create_table(self.db_connection, table_name=self.train_training_info_table, create_table_arg=create_table_arg, df=None)
            du.print_log(f"Table {self.train_training_info_table} created")

    def upsert_feature_training_info_table(self,sql_dict):
        curr_dt = str(dt.datetime.now())
        sql_dict.update({'updated_timestamp':curr_dt})
        update_where_expr = f"validation_partition = '{sql_dict['validation_partition']}' and test_partition = '{sql_dict['test_partition']}' and train_partition = '{sql_dict['train_partition']}' and feature_selection_method ='{sql_dict['feature_selection_method']}' and label_name = '{sql_dict['label_name']}'"
        self.update_insert(sql_dict = sql_dict,
                        table_name = self.train_training_info_table,
                        update_where_expr=update_where_expr)
                
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
            #print(f"Number of mapper column are {len(mapper)} for label {label}")
            #print(mapper)
            self.feature_selection_dict.update({label:[features,ast.literal_eval(mapper)]})
            #print("TYPE",type(ast.literal_eval(mapper)))
            #print(mapper)
    
    def model_train(self,model,db_connection,fold_combinations,feature_names,
                    feature_table_name,label_name,label_mapper,model_params,model_fit_params,prob_theshold_list=[0.8,0.9]):
        
        final_info = {}
        reload_init_model = False
        print(f"Number of features are {len(feature_names)}")
        first_iteration = True
        for col in feature_names:
            print(f"\t {col}")
        i = 0
        for train_combination,validation_combination,test_combination in fold_combinations:
            print("#"*100)
            print(f"Training for : {train_combination}")
            print(f"Validation for : {validation_combination}")
            print(f"Testing for : {test_combination}")
            train_data,train_data_label,validation_data,validation_data_label,test_data,test_data_label,feature_cols= self.get_train_val_data(
                                                                                                    feature_names=feature_names,
                                                                                                    label_name=label_name,
                                                                                                    label_mapper=label_mapper,
                                                                                                    train_filter=train_combination,
                                                                                                    validation_filter=validation_combination,
                                                                                                    test_filter = test_combination)
            
            model_fit_params.update({'X':train_data})
            model_fit_params.update({'y':train_data_label})
            model_fit_params.update({'eval_set':[(validation_data, validation_data_label), (train_data, train_data_label)]})
            model_fit_params.update({'eval_names':['valid','train']})
            print(f"Shape of train data is {train_data.shape}")
            print(f"Shape of train data label is {train_data_label.shape}")
            print(f"Shape of validation data is {validation_data.shape}")
            print(f"Shape of validation label data is {validation_data_label.shape}")
            print(f"Shape of test data is {test_data.shape}")
            print(f"Shape of test data label is {test_data_label.shape}")
            if reload_init_model:
                print(f"Initializing weights with previous model")
                model_fit_params.update({'init_model': model})
                reload_init_model = True
                print(f"Setting reload_init_model to {reload_init_model}")
            if first_iteration:
                feature_importance_values = np.zeros(len(feature_names))
                first_iteration = False
                print(f"Setting first_iteration to {first_iteration}")
            model.fit(**model_fit_params)
            print("MODEL ARCHITECTURE IS ")
            print(model)
            
            #preds_proba = model_prediction(model,test_data,test_data_label,predict_mode='proba',db_connection=None,table_filter=None)
            #preds_predict = model_prediction(model,test_data,test_data_label,predict_mode='predict',db_connection=None,table_filter=None)
            #metric_dict = model_evaluation(features_name,label_name=None,actual_labels=None,test_data=None,db_connection=None,test_table_name=None,table_filter=None,model=None):
            metrics_dict_list = self.model_evaluation(model,
                                                      test_data,
                                                      test_data_label,
                                                      db_connection=db_connection,
                                                      label_name=label_name,
                                                      features_name=None,
                                                      test_table_name=None,
                                                      table_filter=None,
                                                      model_path=None,
                                                      prob_theshold_list=prob_theshold_list)
            
                         
            final_info.update({'label_name':label_name})
            final_info.update({'training_algo':'lightgbm'})
            
            final_info.update({'train_partition':train_combination})
            final_info.update({'test_partition':test_combination})
            final_info.update({'validation_partition':validation_combination})

            #final_info.update({'model_fit_params':model_fit_params})

            final_info.update({'train_data_shape':str(train_data.shape).replace(",","-").replace(" ","")})
            final_info.update({'test_data_shape':str(test_data.shape).replace(",","-").replace(" ","")})
            final_info.update({'validation_data_shape':str(validation_data.shape).replace(",","-").replace(" ","")})
            feature_str="-".join(feature_names).replace("'","\\'").replace(",","\\,")
            model_params = str(model_params).replace("'","\\'").replace(",","\\,")
            #model_params = "dummy"
            #metrics_dict_list = str(metrics_dict_list).replace("'","\\'").replace(",","\\,").replace("{","\\{").replace("}","\\}").replace("[","\\[").replace("]","\\]")
            #model_param = str(model_param).replace("'","\\'")
            #model_param='dummy'
            tc = train_combination.split("_")[-1]
            tst = test_combination.split("_")[-1]
            vc = validation_combination.split("_")[-1]
            metric_file_name = f"{self.train_model_base_path}metric_{label_name}_{tc}_{vc}_{tst}.pkl"

            final_info.update({'feature_names':feature_str})
            final_info.update({'model_params':metrics_dict_list})
            final_info.update({'metrics_dict_list':metric_file_name})

            
            model_file_name = f"{self.train_model_base_path}model_{label_name}_{tc}_{vc}_{tst}.pkl"
            final_info.update({'model_file_name':model_file_name})
            final_info.update({'label_mapper':label_mapper})

            feature_importance_values += model.feature_importances_ 
            
            feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,feature_cols)), columns=['Value','Feature'])
            print(feature_imp)
            final_info.update({'model':'dummy'})
            final_info.update({'feature_table_name':feature_table_name})
            final_info.update({'feature_selection_method':self.feature_selection_method})
            final_info.update({'feature_importance_df':'dummy'})
            
            self.upsert_feature_training_info_table(final_info)
            print(f" BEST MODEL SCORE FOR THIS ITERATION IS : {model.best_score_}")
    
            print("#"*100)
            print(f" MODEL SAVE AT LOCATION : {model_file_name}")
            self.save_pickle_obj(model_file_name,model)
            self.save_pickle_obj(metric_file_name,metrics_dict_list)
            i = i+1
        gc.enable()
        del model
        feature_importance_values = feature_importance_values/i
        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
        fi_file_name = f"{self.train_model_base_path}feature_imp_{label_name}.csv"
        print(feature_importances)
        feature_importances.to_csv(fi_file_name)
        print(f" MODEL OBJECT THRASHED")
        gc.collect()
            
    def get_prev_trained_data(self):
        #a = info['train_partition'].tolist()
        #b = info['test_partition'].tolist()
        #c = info['validation_partition'].tolist()
        sql = f"select * from {self.train_training_info_table}"
        info = ddu.load_table_df(self.db_connection,table_name=None,column_names=None,filter=None,load_sql=sql)
        col_list = list(info['label_name'].unique())
        #ll_dict = {}
        #for col in col_list:
        #    info_tmp = info[info['label_name']==col]
        #    a = info_tmp['train_partition'].tolist()
        #    b = info_tmp['test_partition'].tolist()
        #    c = info_tmp['validation_partition'].tolist()
        #    for i,j,k in zip(a,b,c):
        #        all_dict.update({col:{'train':i,'test':j,'validation':k}})
        return col_list

    def create_model(self,model_params):
        print("Method not implemented in base class")
        model = None
        return model
    
    def train_all_labels(self,model_params,model_fit_params,prob_theshold_list=None,label_name = None,
                         feature_selection_method='featurewiz',force_training_labels=[]):
        self.get_feature_selection_info(feature_selection_method=feature_selection_method)
        self.feature_selection_method = feature_selection_method
        if label_name is not None:
            self.feature_selection_dict = {i:j for i,j in self.feature_selection_dict.items() if i==label_name}
        already_trained_labels = self.get_prev_trained_data()
        if len(force_training_labels)> 0:
            already_trained_labels = [c for c in already_trained_labels if c not in force_training_labels]
        print(f"Already present labels for {'.'.join(already_trained_labels)}")
        self.feature_selection_dict = {i:j for i,j in self.feature_selection_dict.items() if i not in already_trained_labels}
        if prob_theshold_list is not None:
            self.prob_theshold_list = prob_theshold_list
        if len(self.feature_selection_dict) > 0:
            for label_name,feature_map_list in self.feature_selection_dict.items():
                print(f"MODEL TRAINING STARTS FOR {label_name}")
                model = self.create_model(model_params)
                fold_combinations=self.get_label_combination(scramble_all=True,comb_diff=3,select_value=4)
                self.model_train(model,db_connection=self.db_connection,
                                            fold_combinations=fold_combinations,
                                            feature_names=feature_map_list[0],
                                            feature_table_name=self.train_feature_table,
                                            label_name=label_name,
                                            label_mapper=feature_map_list[1],
                                            model_params=model_params,
                                            model_fit_params=model_fit_params,
                                            prob_theshold_list=self.prob_theshold_list  
                                            )
        else:
            print(f"feature_selection_dict is empty {self.feature_selection_dict}. No training will start")

                
    def save_pickle_obj(self,pickle_file_path,pickle_obj):
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(pickle_obj, f)   
    
    def load_pickle_obj(self,pickle_file_path):
        with open(pickle_file_path, 'wb') as f:
            pickle_obj = pickle.load(f)
        return pickle_obj
                            
    def get_train_val_data(self,feature_names,label_name,label_mapper,
                            train_filter,validation_filter,test_filter,
                            if_return_df = False):
        #train_data = ddu.load_table_df(db_connection,column_names=feature_names,table_name=table_name,filter=f"time_split = '{train_filter}'")
        #train_data_label = ddu.load_table_df(db_connection,column_names=label_name,table_name=table_name,filter=f"time_split = '{train_filter}'")
        #validation_data = ddu.load_table_df(db_connection,column_names=feature_names,table_name=table_name,filter=f"time_split = '{validation_filter}'")
        #validation_data_label = ddu.load_table_df(db_connection,column_names=label_name,table_name=table_name,filter=f"time_split = '{validation_filter}'")

        sql = f"select {','.join(feature_names)},{label_name} from {self.train_feature_table} where time_split = '{train_filter}' and {label_name} != 'unknown'"
        print("Training data extract sql is :")
        print(sql)
        train_data = ddu.load_table_df(self.db_connection,table_name=None,column_names=None,filter=None,load_sql=sql)
        train_data = train_data.dropna()
        print("Train Data label distibution is :")
        print(train_data[label_name].value_counts())
        train_data_label = train_data[[label_name]]
        train_data = train_data[feature_names]
        #sql = f"select {label} from banknifty.train_feature where time_split = '{train_filter}'  and label != 'unknown'"
        #train_data_label = ddu.load_table_df(con,table_name=None,column_names=None,filter=None,load_sql=sql)
        
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
        #sql = f"select {label} from banknifty.train_feature where time_split = '{validation_filter}'  and label != 'unknown'"
        #validation_data_label = ddu.load_table_df(con,table_name=None,column_names=None,filter=None,load_sql=sql)
        
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
        #sql = f"select {label} from banknifty.train_feature where time_split = '{validation_filter}'  and label != 'unknown'"
        #validation_data_label = ddu.load_table_df(con,table_name=None,column_names=None,filter=None,load_sql=sql)
        
        test_data_label[label_name] = test_data_label[label_name].map(label_mapper)
        
        #sql = f"select {label} from banknifty.train_feature where time_split = '{test_filter}'  and label != 'unknown'"
        #test_data_label = ddu.load_table_df(con,table_name=None,column_names=None,filter=None,load_sql=sql)
        #test_data_label[label] = test_data_label[label].map({'neutral':0,'call':1,'put':2})

        print('Training Data Shape: ', train_data.shape)
        print('Validation Data Shape: ', validation_data.shape)
        print('Testing Data Shape: ', test_data.shape)
        feature_cols = train_data.columns.tolist()
        if if_return_df is False:
            train_data = np.array(train_data)
            train_data_label = np.array(train_data_label)
            validation_data = np.array(validation_data)
            validation_data_label = np.array(validation_data_label)
            test_data = np.array(test_data)
            test_data_label = np.array(test_data_label)
        return train_data,train_data_label,validation_data,validation_data_label,test_data,test_data_label,feature_cols

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
        sql = f"select distinct time_split from banknifty.train_feature"
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
        #for i in range(len(train_split)):
        #  try:
        #    comb_list.append([train_split[i],validation_split[i],test_split[i]])
        #  except Exception as e:
        #    print(f"Error {e}")
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
                         db_connection=None,
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
            test_data = ddu.load_table_df(connection=db_connection,
                                            column_names = features_name,
                                            table_name=test_table_name,
                                            filter=table_filter)
            test_data = np.array(test_data)
            actual_labels = ddu.load_table_df(connection=db_connection,
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