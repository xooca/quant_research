
from sklearn.pipeline import Pipeline
import pickle
import logging
import data.features.data_engine as de
import pandas as pd
import numpy as np
# import data.data_config as dc
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
import data.utils.data_utils as du
import duckdb


class pipelines:
    def __init__(self, master_config_path, master_config_name, db_conection, 
                 database_path=None,
                 train_feature_table=None,
                 train_feature_info_table=None,
                 techindicator1=True,
                 techindicator2=True,
                 techindicator3=True,
                 time_splitter=True,
                 column_unstable=True,
                 label_creation=True,
                 update_unstable=True,
                 verbose=True):
        
        self.database_path = database_path
        if self.database_path is None:
            self.db_conection = db_conection
        else:
            self.db_conection = duckdb.connect(database=self.database_path , read_only=False)
        self.feature_mart = de.feature_mart(master_config_path=master_config_path,
                                            master_config_name=master_config_name,
                                            db_conection=self.db_conection,database_path= self.database_path,
                                            train_feature_table=train_feature_table,
                                            train_feature_info_table=train_feature_info_table,
                                            verbose=verbose)
        self.techindicator1 = techindicator1
        self.techindicator2 = techindicator2
        self.techindicator3 = techindicator3
        self.time_splitter = time_splitter
        self.column_unstable = column_unstable
        self.label_creation = label_creation
        self.update_unstable = update_unstable
        
        
    def save_check_point(self,end_conection=False):
        if self.database_path is not None:
            self.feature_mart.db_conection.close()
            if not end_conection:
                du.print_log(f"Checkpointing successful")
                self.feature_mart.db_conection = duckdb.connect(database=self.database_path , read_only=False)
            else:
                du.print_log(f"Connection closed")
        else:
            du.print_log(f"Checkpointing not possible as database_path is {self.database_path}")

    def pipeline_definitions(self):
        
        # Creation of labels
        if self.label_creation:
            for label in ['label_generator_4class_50points', 'label_generator_3class_50points_no_neutral','label_generator_3class_20points_no_neutral']:
                for sht in [-15,-30,-45,-60]:
                    label_creator_args = {'freq': '1min', 'shift': sht,'shift_column': 'close', 'generator_function_name': label}
                    self.feature_mart.label_creator(
                        label_creator_args, tmpdf=None, return_df=False)
            self.save_check_point()
        
        self.save_check_point(end_conection=True)