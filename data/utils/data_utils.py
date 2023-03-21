import data.utils.duckdb_utils as ddu
from sklearn.pipeline import Pipeline
import data.features.data_engine as de
import pickle
import logging
import requests
import omegaconf
# import data.data_config as dc
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
import hydra
from omegaconf import OmegaConf
import gc
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
import warnings
import importlib
from sklearn.model_selection import train_test_split
import zipfile
import fnmatch
import os
from pathlib import Path
import string
import re
import gc
from config.common.config import Config, DefineConfig
from datetime import datetime

warnings.filterwarnings('ignore')

os.getcwd()
pd.options.display.max_columns = None


def downcast_df_float_columns(df):
    list_of_columns = list(df.select_dtypes(include=["float64"]).columns)
    if len(list_of_columns) >= 1:
        for col in list_of_columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
    else:
        print("no columns to downcast")
    gc.collect()
    return df


def downcast_df_int_columns(df):
    list_of_columns = list(df.select_dtypes(
        include=["int32", "int64"]).columns)
    if len(list_of_columns) >= 1:
        for col in list_of_columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
    else:
        print("no columns to downcast")
    gc.collect()
    return df


def reduce_mem_usage_v1(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    for var in df.columns.tolist():
        if df[var].dtype != object:
            maxi = df[var].max()
            if maxi < 255:
                df[var] = df[var].astype(np.uint8)
                print(var, "converted to uint8")
            elif maxi < 65535:
                df[var] = df[var].astype(np.uint16)
                print(var, "converted to uint16")
            elif maxi < 4294967295:
                df[var] = df[var].astype(np.uint32)
                print(var, "converted to uint32")
            else:
                df[var] = df[var].astype(np.uint64)
                print(var, "converted to uint64")
    mem_usg = df.memory_usage().sum() / 1024**2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100*mem_usg/start_mem_usg, "% of the initial size")
    return df


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            if str(col_type)[:3] == 'int':
                df[col] = df[col].astype(np.int32)
            if str(col_type)[:5] == 'float':
                df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))
    return df


def reduce_mem_usage_v2(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))
    return df


def remove_specialchars(text):
    text_re = re.escape(string.punctuation)
    return re.sub(r'['+text_re+']', '', text)


def nullcolumns(df):
    t = pd.DataFrame(df[df.columns[df.isnull().any()]
                        ].isnull().sum()).reset_index()
    t.columns = ['colname', 'nullcnt']
    t = t.sort_values(by='nullcnt', ascending=False)
    return t


def checknans(df, threshold=100):
    nan_cols = []
    for col in df.columns.tolist():
        if sum(pd.isna(df[col])) > threshold:
            print(f"{col}.... {sum(np.isnan(df[col]))}")
            nan_cols.append(col)
    return nan_cols


def initialize_config(overrides, version_base=None, config_path="../config"):
    initialize(config_path=config_path)
    print(overrides, config_path)
    dc = compose(overrides=overrides)
    return dc


def print_log(log, using_print='print'):
    if using_print == 'print':
        print(log)
    else:
        logging.info(log)


def check_and_create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def convert_df_to_timeseries(df, convert_date_fmt=False):
    df['date'] = df['date'].astype(str)
    df['time'] = df['time'].astype(str)
    if convert_date_fmt:
        df['date'] = df['date'].apply(lambda s: s[:4]+"-" + s[4:6]+"-"+s[6:])
        df['time'] = df['time'].apply(lambda s: s+':'+'00')
    df['timestamp'] = df['date'] + ' ' + df['time']
    df = df.sort_values(by='timestamp')
    # df.index = df['date_time']
    
    df = df[['timestamp', 'ticker', 'open', 'high', 'low', 'close']]
    return df


def load_object(object_path):
    with open(object_path, 'rb') as handle:
        return_obj = pickle.load(handle)
        print_log(f"Object loaded")
    return return_obj


def save_object(object_path, obj):
    with open(object_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print_log(f"Object saved at location {object_path}")


class initial_data_setup(DefineConfig):
    def __init__(self, master_config_path, master_config_name, db_conection):
        DefineConfig.__init__(self, master_config_path, master_config_name)
        self.db_conection = db_conection
        self.completed_files = {}

    def unzip_folders(self):
        self.f_name_list = []
        if not ddu.check_if_table_exists(self.db_conection, table_name=self.zip_files_table):
            create_table_arg = {
                'replace': True, 'table_column_arg': 'timestamp TIMESTAMP,file_name VARCHAR PRIMARY KEY,status VARCHAR'}
            ddu.create_table(self.db_conection, table_name=self.zip_files_table,
                             create_table_arg=create_table_arg, df=None)

        for root, dirs, files in os.walk(self.raw_data_input_path):
            for filename in fnmatch.filter(files, self.zip_file_pattern):
                f_name = os.path.join(root, filename)
                if zipfile.is_zipfile(f_name):
                    try:
                        n_file = os.path.join(
                            root, os.path.splitext(filename)[0])
                        zipfile.ZipFile(f_name).extractall(n_file)
                        print_log(
                            f"File saved at location {n_file}", self.using_print)
                        os.remove(f_name)
                        self.f_name_list.append(f_name)
                        print_log(f"File {f_name} removed", self.using_print)
                        ct = datetime.now()
                        insert_arg = {
                            'insert_values': f"'{ct}','{f_name}','unzipped'"}
                        ddu.insert_data(
                            self.db_conection, table_name=self.zip_files_table, insert_arg=insert_arg, df=None)
                    except Exception as e1:
                        print_log(
                            f"File {f_name} is not unzipped", self.using_print)
                        print_log(
                            f"Error encountered is {e1}", self.using_print)
                        ct = datetime.now()
                        insert_arg = {
                            'insert_values': f"'{ct}','{f_name}','{e1}'"}
                        ddu.insert_data(
                            self.db_conection, table_name=self.zip_files_table, insert_arg=insert_arg, df=None)
                else:
                    print_log(f"File {f_name} is not unzipped",
                              self.using_print)

    def create_dataset(self, reload_all=True):
        self.loaded_source_files_table_df = pd.DataFrame()
        already_loaded_files = set()
        print_log(f'Source data path is { self.source_data}')
        if not os.path.exists(self.source_data):
            os.makedirs(self.source_data)
            print_log(f'Created folder {self.source_data}', self.using_print)

        print_log(f'Data save path is { self.raw_data_save_path}')
        if not ddu.check_if_table_exists(self.db_conection, table_name=self.ohlc_raw_data_table):
            create_table_arg = {
                'replace': True, 'table_column_arg': 'timestamp TIMESTAMP,ticker VARCHAR,open INTEGER,high INTEGER,low INTEGER,close INTEGER'}
            ddu.create_table(self.db_conection, table_name=self.ohlc_raw_data_table,
                             create_table_arg=create_table_arg, df=None)

        try:
            self.loaded_source_files_table_df = ddu.load_table_df(
                self.db_conection, table_name=self.loaded_source_files_table)
            print_log(
                f"{len(self.loaded_source_files_table_df)} data present in loaded_source_files_table_df table", self.using_print)
        except Exception as e1:
            create_table_arg = {
                'replace': True, 'table_column_arg': 'timestamp TIMESTAMP PRIMARY KEY,file_name VARCHAR,size INTEGER, status VARCHAR'}
            ddu.create_table(self.db_conection, table_name=self.loaded_source_files_table,
                             create_table_arg=create_table_arg, df=None)
        if len(self.loaded_source_files_table_df) > 0 and reload_all is False:
            already_loaded_files = set(
                self.loaded_source_files_table_df['file_name'].to_list())
            print_log(
                f"{len(already_loaded_files)} files already loaded", self.using_print)
        for root, dirs, files in os.walk(self.raw_data_input_path):
            for filename in fnmatch.filter(files, self.data_pattern):
                try:
                    f_name = Path(os.path.join(root, filename))
                    if str(f_name) not in already_loaded_files:
                        tmp_df = pd.read_csv(f_name, header=None)
                        tmp_df = tmp_df.loc[:, 0:6]
                        tmp_df.columns = self.initial_columns

                        tmp_df = convert_df_to_timeseries(
                            tmp_df, convert_date_fmt=True)
                        tmp_df = tmp_df.drop_duplicates(subset='timestamp',keep='first')
                        print_log(
                            f"Data loaded from file {f_name} and has shape {tmp_df.shape}", self.using_print)
                        ddu.insert_data(
                            self.db_conection, table_name=self.ohlc_raw_data_table, insert_arg={}, df=tmp_df)
                        ct = datetime.now()
                        insert_arg = {
                            'insert_values': f"'{ct}','{f_name}','{len(tmp_df)}','loaded'"}
                        ddu.insert_data(
                            self.db_conection, table_name=self.loaded_source_files_table, insert_arg=insert_arg, df=None)
                except Exception as e:
                    print_log(
                        f"Error while writing file to {self.loaded_source_files_table} : {e}", self.using_print)
                    print_log(
                        f"Unable to write {f_name} to the table", self.using_print)
                    ct = datetime.now()
                    e = str(e).replace("'", "").replace('"', '')
                    insert_arg = {
                        'insert_values': f"'{ct}','{f_name}','{len(tmp_df)}','{e}'"}
                    ddu.insert_data(
                        self.db_conection, table_name=self.loaded_source_files_table, insert_arg=insert_arg, df=None)


class execute_data_pipeline:
    def __init__(self, master_config_path, master_config_name, db_conection, database_path,feature_spec_name, train_feature_table=None,train_feature_info_table=None,
                 techindicator1=True,techindicator2=True,techindicator3=True,time_splitter=True,column_unstable=True,
                 label_creation=True,update_unstable=True,load_tmp=False,verbose=True):
        # DefineConfig.__init__(self,master_config_path,master_config_name)
        #self.db_conection = db_conection
        self.feature_spec_name = feature_spec_name
        print_log(f"Feature spec file is {self.feature_spec_name}")
        self.feature_spec = importlib.import_module(
            f"{self.feature_spec_name}")
        self.feature_pipeline = self.feature_spec.pipelines(master_config_path, 
                                                            master_config_name, 
                                                            db_conection,
                                                            database_path,
                                                            train_feature_table,
                                                            train_feature_info_table,
                                                            techindicator1,
                                                            techindicator2,
                                                            techindicator3,
                                                            time_splitter,
                                                            column_unstable,
                                                            label_creation,
                                                            update_unstable,
                                                            load_tmp,
                                                            verbose)

    def run_pipelines(self):
        self.feature_pipeline.pipeline_definitions()

class execute_feature_selection_pipeline:
    def __init__(self, 
                 master_config_path, 
                 master_config_name, 
                 db_connection, 
                 database_path,
                 feature_selection_spec_name, 
                 train_feature_table=None,
                 train_feature_selection_table = None,
                 train_feature_info_table=None,
                 filter_out_cols=None,
                 ignore_cols = None,
                 verbose=True):

        
        self.feature_selection_spec_name = feature_selection_spec_name
        print_log(f"Feature selection spec file is {self.feature_selection_spec_name}")
        self.feature_selection_spec = importlib.import_module(f"{self.feature_selection_spec_name}")
        self.feature_selection_pipeline = self.feature_selection_spec.feature_selection(master_config_path=master_config_path, 
                                                                              master_config_name=master_config_name, 
                                                                              db_connection=db_connection,
                                                                              database_path=database_path,
                                                                              train_feature_table=train_feature_table,
                                                                              train_feature_selection_table=train_feature_selection_table,
                                                                              train_feature_info_table=train_feature_info_table,
                                                                              filter_out_cols=filter_out_cols,ignore_cols=ignore_cols,
                                                                              verbose=verbose)

    def run_pipelines(self,forced_labels=[],split_selection_limit=1):
        self.feature_selection_pipeline.run_feature_selection(forced_labels=forced_labels,split_selection_limit=split_selection_limit)
        

class read_data_api:
    def __init__(self, master_config):
        master_config = dict(master_config['master']['model'])
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.config = initialize_config(**master_config)
        self.using_print = True if self.config.data.generic.verbose_type == 'print' else False
        print(self.config)
        self.base_url = self.config.common.endpoint.base_url
        self.endpoint_details = self.config.common.endpoint_details
        self.read_action = self.config.common.endpoint.read_action

    def create_url(self, url=None):
        if url is None:
            endpoint_details = dict(self.endpoint_details)
            sub_url = ''
            for i, j in endpoint_details.items():
                if j != 'None':
                    sub_url += f"{i}={j}&"
            self.url = f"{self.base_url}{sub_url}"
            self.url = self.url[:-1]
        else:
            self.url = url
        print_log(f"API URL is {self.url}", self.using_print)

    def call_api(self, url=None, read_action=None):
        self.create_url(url)
        if read_action is None:
            if self.read_action == 'json':
                r = requests.get(self.url)
                data = r.json()
            elif self.read_action == 'csv':
                data = pd.read_csv(self.url)
            else:
                print_log(
                    f"Invalid read action: {self.read_action}", self.using_print)
                data = None
        return data


class datacleaner:
    def __init__(self, df, targetcol, id_cols=None, cat_threshold=100):
        self.df_train = df
        self.target = targetcol
        self.id = id_cols
        self.dfcolumns = self.df_train.columns.tolist()
        self.dfcolumns_nottarget = [
            col for col in self.dfcolumns if col != self.target]
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        self.num_cols = self.df_train.select_dtypes(
            include=numerics).columns.tolist()
        self.num_cols = [
            col for col in self.num_cols if col not in [self.target, self.id]]
        self.non_num_cols = [
            col for col in self.dfcolumns if col not in self.num_cols + [self.target, self.id]]
        self.rejectcols = []
        self.retainedcols = []
        self.catcols = {}
        self.noncatcols = {}
        self.catcols_list = []
        self.noncatcols_list = []
        self.hightarge_corr_col = []
        self.threshold = cat_threshold

    def normalize_column_name(self, col_name):
        col_name = str(col_name)
        col_name = col_name.lower()
        col_name = col_name.strip()
        col_name = col_name.replace(' ', '_')
        col_name = col_name.replace(r"[^a-zA-Z\d\_]+", "")
        return col_name

    def normalize_metadata(self, tmpdf):
        self.df_train = tmpdf
        self.target = self.normalize_column_name(self.target)
        self.id = self.normalize_column_name(self.id)
        self.target = self.normalize_column_name(self.target)
        self.dfcolumns = [self.normalize_column_name(
            col) for col in self.dfcolumns]
        self.dfcolumns_nottarget = [self.normalize_column_name(
            col) for col in self.dfcolumns_nottarget]
        self.num_cols = [self.normalize_column_name(
            col) for col in self.num_cols]
        self.non_num_cols = [self.normalize_column_name(
            col) for col in self.non_num_cols]

    def clean_column_name(self):
        def clean_column_name_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    tmpdf.columns = [str(x).lower()
                                     for x in tmpdf.columns.tolist()]
                    tmpdf.columns = tmpdf.columns.str.strip()
                    tmpdf.columns = tmpdf.columns.str.replace(' ', '_')
                    tmpdf.columns = tmpdf.columns.str.replace(
                        r"[^a-zA-Z\d\_]+", "")
                    self.normalize_metadata(tmpdf)
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])

            return wrapper

        return clean_column_name_lvl

    def reject_null_cols(self, null_threshold):
        def reject_null_cols_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    for col in tmpdf:
                        null_count = sum(tmpdf[col].astype(str).isnull())
                        if null_count > 0:
                            percent_val = null_count/tmpdf[col].shape[0]
                            if percent_val > null_threshold:
                                self.rejectcols.append(col)
                    self.retainedcols = [
                        col for col in tmpdf.columns.tolist() if col not in self.rejectcols]
                    print(
                        f"INFO : {str(datetime.now())} : Number of rejected columns {len(self.rejectcols)}")
                    print(
                        f"INFO : {str(datetime.now())} : Number of retained columns {len(self.retainedcols)}")
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])
            return wrapper

        return reject_null_cols_lvl

    def standardize_stratified(self, auto_standard=True, includestandcols=[], for_columns=[]):
        def standardize_stratified_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    if len(for_columns) == 0:
                        if auto_standard:
                            stand_cols = list(
                                set(self.num_cols + includestandcols))
                        else:
                            stand_cols = self.num_cols
                    else:
                        stand_cols = for_columns.copy()
                    for col in tmpdf:
                        if col in stand_cols:
                            tmpdf[col] = tmpdf[col].astype(np.float)
                            tmpdf[col] = tmpdf[col].replace(np.inf, 0.0)
                            if tmpdf[col].mean() > 1000:
                                scaler = MinMaxScaler(feature_range=(0, 10))
                                tmpdf[col] = scaler.fit_transform(
                                    np.asarray(tmpdf[col]).reshape(-1, 1))
                            elif tmpdf[col].mean() > 100:
                                scaler = MinMaxScaler(feature_range=(0, 5))
                                # print(col)
                                tmpdf[col] = scaler.fit_transform(
                                    np.asarray(tmpdf[col]).reshape(-1, 1))
                            elif tmpdf[col].mean() > 10:
                                scaler = MinMaxScaler(feature_range=(0, 2))
                                # print(col)
                                tmpdf[col] = scaler.fit_transform(
                                    np.asarray(tmpdf[col]).reshape(-1, 1))
                            else:
                                scaler = MinMaxScaler(feature_range=(0, 1))
                                tmpdf[col] = scaler.fit_transform(
                                    np.asarray(tmpdf[col]).reshape(-1, 1))
                            print("INFO : " + str(datetime.now()) +
                                  ' : ' + 'Column ' + col + 'is standardized')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])

            return wrapper

        return standardize_stratified_lvl

    def featurization(self, cat_coltype=False):
        def featurization_lvl1(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    self.refresh_cat_noncat_cols_fn(tmpdf, self.threshold)
                    if cat_coltype:
                        column_list = self.catcols_list
                    else:
                        column_list = self.noncatcols_list
                    print("INFO : " + str(datetime.now()) + ' : ' + 'Shape of dataframe before featurization ' + str(
                        tmpdf.shape))
                    for col in column_list:
                        tmpdf[col + '_minus_mean'] = tmpdf[col] - \
                            np.mean(tmpdf[col])
                        tmpdf[col + '_minus_mean'] = tmpdf[col +
                                                           '_minus_mean'].astype(np.float32)
                        tmpdf[col + '_minus_max'] = tmpdf[col] - \
                            np.max(tmpdf[col])
                        tmpdf[col + '_minus_max'] = tmpdf[col +
                                                          '_minus_max'].astype(np.float32)
                        tmpdf[col + '_minus_min'] = tmpdf[col] - \
                            np.min(tmpdf[col])
                        tmpdf[col + '_minus_min'] = tmpdf[col +
                                                          '_minus_min'].astype(np.float32)
                    print("INFO : " + str(datetime.now()) + ' : ' + 'Shape of dataframe after featurization ' + str(
                        tmpdf.shape))
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])

            return wrapper

        return featurization_lvl1

    def feature_importance(self, dfforimp, tobepredicted, modelname, featurelimit=0):
        colname = [col for col in dfforimp.columns.tolist() if col !=
                   tobepredicted]
        X = dfforimp[colname]
        y = dfforimp[tobepredicted]
        # print(modelname)
        # t =''
        if modelname == 'rfclassifier':
            model = RandomForestClassifier(n_estimators=100, random_state=10)
        elif modelname == 'rfregressor':
            model = RandomForestRegressor(n_estimators=100, random_state=10)
        elif modelname == 'lgbmclassifier':
            model = lgb.LGBMClassifier(
                n_estimators=1000, learning_rate=0.05, verbose=-1)
        elif modelname == 'lgbmregressor':
            # print('yes')
            model = lgb.LGBMRegressor(
                n_estimators=1000, learning_rate=0.05, verbose=-1)
        else:
            print("Please specify the modelname")
        model.fit(X, y)
        feature_names = X.columns
        feature_importances = pd.DataFrame(
            {'feature': feature_names, 'importance': model.feature_importances_})
        feature_importances = feature_importances.sort_values(
            by=['importance'], ascending=False).reset_index()
        feature_importances = feature_importances[['feature', 'importance']]
        if featurelimit == 0:
            return feature_importances
        else:
            return feature_importances[:featurelimit]

    def importantfeatures(self, dfforimp, tobepredicted, modelname, skipcols=[], featurelimit=0):
        # print(modelname)
        f_imp = self.feature_importance(
            dfforimp, tobepredicted, modelname, featurelimit)
        allimpcols = list(f_imp['feature'])
        stuff = []
        for col in allimpcols:
            for skipcol in skipcols:
                if col != skipcol:
                    stuff.append(col)
                else:
                    pass
        return stuff, f_imp

    def convertdatatypes(self, cat_threshold=100):
        def convertdatatypes_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    self.dfcolumns_nottarget = [
                        col for col in tmpdf.columns.tolist() if col != self.target]
                    for c in self.dfcolumns_nottarget:
                        col_dtype = tmpdf[c].dtype
                        if (col_dtype == 'object') and (tmpdf[c].nunique() < cat_threshold):
                            tmpdf[c] = tmpdf[c].astype('category')
                        elif (col_dtype in ['int64', 'int32']) and (tmpdf[c].nunique() < cat_threshold):
                            tmpdf[c] = tmpdf[c].astype('category')
                        elif col_dtype in ['float64']:
                            tmpdf[c] = tmpdf[c].astype(np.float32)
                        elif col_dtype in ['int64',]:
                            tmpdf[c] = tmpdf[c].astype(np.int32)
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])

            return wrapper

        return convertdatatypes_lvl

    def ohe_on_column(self, columns=None, drop_converted_col=True, refresh_cols=True):
        def converttodummies_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    if columns is None:
                        if refresh_cols:
                            self.refresh_cat_noncat_cols_fn(
                                tmpdf, self.threshold)
                        column_list = self.catcols_list
                    else:
                        column_list = columns
                    for col in column_list:
                        dummy = pd.get_dummies(tmpdf[col])
                        dummy.columns = [
                            col.lower() + '_' + str(x).lower().strip() + '_dums' for x in dummy.columns]
                        tmpdf = pd.concat([tmpdf, dummy], axis=1)
                        if drop_converted_col:
                            tmpdf = tmpdf.drop(col, axis=1)
                        print("INFO : " + str(datetime.now()) + ' : ' +
                              'Column ' + col + ' converted to dummies')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])

            return wrapper

        return converttodummies_lvl

    def remove_collinear(self, th=0.95):
        def converttodummies_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    print("INFO : " + str(datetime.now()) + ' : ' + 'Shape of dataframe before collinear drop ' + str(
                        tmpdf.shape))
                    corr_matrix = tmpdf.corr().abs()
                    upper = corr_matrix.where(
                        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                    to_drop = [column for column in upper.columns if any(
                        upper[column] > th)]
                    tmpdf = tmpdf.drop(to_drop, axis=1)
                    print("INFO : " + str(datetime.now()) + ' : ' + 'Shape of dataframe after collinear drop ' + str(
                        tmpdf.shape))
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])

            return wrapper

        return converttodummies_lvl

    def high_coor_target_column(self, targetcol='y', th=0.5):
        def high_coor_target_column_lvl1(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    print("INFO : " + str(
                        datetime.now()) + ' : ' + 'Shape of dataframe before retaining only highly corelated col with target ' + str(
                        tmpdf.shape))
                    cols = [col for col in tmpdf.columns.tolist() if col !=
                            targetcol]
                    for col in cols:
                        tmpdcorr = tmpdf[col].corr(tmpdf[targetcol])
                        if tmpdcorr > th:
                            self.hightarge_corr_col.append(col)
                    cols = self.hightarge_corr_col + [targetcol]
                    tmpdf = tmpdf[cols]
                    print("INFO : " + str(
                        datetime.now()) + ' : ' + 'Shape of dataframe after retaining only highly corelated col with target ' + str(
                        tmpdf.shape))
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])

            return wrapper

        return high_coor_target_column_lvl1

    def apply_agg_diff(self, aggfunc='median', quantile_val=0.5, columns=[]):
        def apply_agg_diff_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    for col in columns:
                        if aggfunc == 'median':
                            diff_val = np.median(tmpdf[col])
                            tmpdf[f'{col}_mediandiff'] = tmpdf[col].apply(
                                lambda x: x - diff_val)
                        elif aggfunc == 'mean':
                            diff_val = np.mean(tmpdf[col])
                            tmpdf[f'{col}_meandiff'] = tmpdf[col].apply(
                                lambda x: x - diff_val)
                        elif aggfunc == 'min':
                            diff_val = np.min(tmpdf[col])
                            tmpdf[f'{col}_mindiff'] = tmpdf[col].apply(
                                lambda x: x - diff_val)
                        elif aggfunc == 'max':
                            diff_val = np.max(tmpdf[col])
                            tmpdf[f'{col}_maxdiff'] = tmpdf[col].apply(
                                lambda x: x - diff_val)
                        elif aggfunc == 'max':
                            diff_val = np.max(tmpdf[col])
                            tmpdf[f'{col}_maxdiff'] = tmpdf[col].apply(
                                lambda x: x - diff_val)
                        else:
                            diff_val = np.quantile(tmpdf[col], quantile_val)
                            tmpdf[f'{col}_q{quantile_val}diff'] = tmpdf[col].apply(
                                lambda x: x - diff_val)
                        print("INFO : " + str(datetime.now()) + ' : ' +
                              'Column ' + col + f' converted using {aggfunc}')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])

            return wrapper

        return apply_agg_diff_lvl

    def logtransform(self, logtransform_col=[]):
        def logtransform_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    for col in logtransform_col:
                        tmpdf[col] = tmpdf[col].apply(
                            lambda x: np.log(x) if x != 0 else 0)
                        print("INFO : " + str(
                            datetime.now()) + ' : ' + 'Column ' + col + ' converted to corresponding log using formula: log(x)')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])

            return wrapper

        return logtransform_lvl

    def binary_target_encode(self, encode_save_path, encoding_cols=[], load_previous=False):
        def target_encode_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    for col in encoding_cols:
                        if load_previous:
                            df = pd.read_csv(f'{encode_save_path}{col}.csv')
                        else:

                            df = tmpdf[[col, self.target]].groupby([col]).sum(
                            ).reset_index().sort_values(by=self.target, ascending=False)
                            df.columns = [col, f'{col}_tgt_enc']
                            df.to_csv(
                                f'{encode_save_path}{col}.csv', index=False)
                        tmpdf = pd.merge(tmpdf, df, on=col, how='left')
                        tmpdf[col] = tmpdf[col].astype(np.float)
                        tmpdf[col] = tmpdf[col].fillna(0)
                        print("INFO : " + str(datetime.now()) + ' : ' +
                              'Column ' + col + ' target encoded')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])
            return wrapper
        return target_encode_lvl

    def binary_target_ratio_encode(self, encode_save_path, encoding_cols=[], load_previous=False):
        def target_ratio_encode_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    for col in encoding_cols:
                        if load_previous:
                            df = pd.read_csv(
                                f'{encode_save_path}{col}_tgt_ratio_enc.csv')
                        else:
                            x = tmpdf[[col, self.target]].groupby(
                                [col]).sum().reset_index()
                            y = tmpdf[[col, self.target]].groupby(
                                [col]).count().reset_index()
                            df = pd.merge(x, y, on=col)
                            df[f'{col}_tgt_ratio_enc'] = df[f'{self.target}_x'] / \
                                df[f'{self.target}_y']
                            df = df[[col, f'{col}_tgt_ratio_enc']]
                            df.to_csv(
                                f'{encode_save_path}{col}_tgt_ratio_enc.csv', index=False)
                        tmpdf = pd.merge(tmpdf, df, on=col, how='left')
                        tmpdf[col] = tmpdf[col].astype(np.float)
                        tmpdf[col] = tmpdf[col].fillna(0)
                        print("INFO : " + str(datetime.now()) + ' : ' +
                              'Column ' + col + ' target encoded')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])
            return wrapper
        return target_ratio_encode_lvl

    def refresh_cat_noncat_cols_fn(self, tmpdf, cat_threshold=100):
        try:
            self.catcols = {}
            self.catcols_list = []
            self.noncatcols = {}
            self.noncatcols_list = []
            self.dfcolumns = tmpdf.columns.tolist()
            self.dfcolumns_nottarget = [
                col for col in self.dfcolumns if col != self.target]
            for col in self.dfcolumns_nottarget:
                col_unique_cnt = tmpdf[col].nunique()
                if (col_unique_cnt < cat_threshold) and (
                        (tmpdf[col].dtype != 'float32') and (tmpdf[col].dtype != 'float64')):
                    self.catcols.update({col: col_unique_cnt})
                    self.catcols_list.append(col)
                else:
                    self.noncatcols.update({col: col_unique_cnt})
                    self.noncatcols_list.append(col)
        except:
            sys.exit('ERROR : ' + str(datetime.now()) +
                     ' : ' + sys.exc_info()[1])

    def refresh_cat_noncat_cols(self, cat_threshold):
        def refresh_cat_noncat_cols_lvl1(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    self.catcols = {}
                    self.catcols_list = []
                    self.noncatcols = {}
                    self.noncatcols_list = []
                    self.dfcolumns = tmpdf.columns.tolist()
                    self.dfcolumns_nottarget = [
                        col for col in self.dfcolumns if col != self.target]
                    for col in self.dfcolumns_nottarget:
                        col_unique_cnt = tmpdf[col].nunique()
                        if (col_unique_cnt < cat_threshold) and (
                                (tmpdf[col].dtype != 'float32') and (tmpdf[col].dtype != 'float64')):
                            self.catcols.update({col: col_unique_cnt})
                            self.catcols_list.append(col)
                        else:
                            self.noncatcols.update({col: col_unique_cnt})
                            self.noncatcols_list.append(col)
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])

            return wrapper
        return refresh_cat_noncat_cols_lvl1
