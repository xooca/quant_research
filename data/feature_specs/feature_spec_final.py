
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
            for label in ['label_generator_4class', 'label_generator_4class_mod1','label_generator_4class_mod2','label_generator_7class']:
                for sht in [-15,-30,-45,-60]:
                    label_creator_args = {'freq': '1min', 'shift': sht,'shift_column': 'close', 'generator_function_name': label}
                    self.feature_mart.label_creator(
                        label_creator_args, tmpdf=None, return_df=False)
            self.save_check_point()
        
        
        # Creation of technical indicators using pandata
        if self.techindicator1 == True:
            create_technical_indicator_using_pandasta_args = {'exclude': ["jma",
                                                                        "pvo", "vwap", "vwma", "ad", 
                                                                        "adosc", "aobv", "cmf", "efi",
                                                                        "eom", "kvo", "mfi", "nvi", "obv",
                                                                        "pvi", "pvol", "pvr", "pvt"]}
            self.feature_mart.create_technical_indicator_using_pandasta(create_technical_indicator_using_pandasta_args, 
                                                                        tmpdf=None, return_df=False)
            self.save_check_point()
        
        # Creation of technical indicators using signals
        if self.techindicator2 == True:
            create_technical_indicator_using_signals_args = {'method_type': ['volumn_', 'volatile_', 'transform_', 
                                                                            'cycle_', 'pattern_', 'stats_', 'math_', 'overlap_']}
            self.feature_mart.create_technical_indicator_using_signals(create_technical_indicator_using_signals_args, 
                                                                    tmpdf=None, return_df=False)
            self.save_check_point()
        
        # Creation of technical indicators using ta
        if self.techindicator3 == True:
            create_technical_indicator_using_ta_args = {'open': 'open', 'high': 'high', 'low': 'low', 
                                                        'close': 'close','vectorized': True,
                                                        'fillna': False, 'colprefix': 'ta', 
                                                        'volume_ta': False, 'trend_ta': True, 
                                                        'momentum_ta': True, 'others_ta': True, 'verbose': True}
            self.feature_mart.create_technical_indicator_using_ta(create_technical_indicator_using_ta_args, 
                                                                tmpdf=None, return_df=False)
            self.save_check_point()
        
        # Creation of last_tick_greater_values_count features
        for tick in [15, 45, 90, 150,300,500]:
            for col in ['close', 'high', 'low']:
                last_tick_greater_values_count_args = {'column': col, 'last_ticks': tick}
                self.feature_mart.last_tick_greater_values_count(
                    last_tick_greater_values_count_args, tmpdf=None, return_df=False)
        self.save_check_point()

        # Creation of price_last_tick_breach_count features
        for tick in [15, 45, 90, 150,300,500]:
            for col in ['close', 'high', 'low']:
                price_last_tick_breach_count_args = {'column_name': col, 'breach_type': 'morethan', 'last_ticks': tick}
                self.feature_mart.price_last_tick_breach_count(price_last_tick_breach_count_args, 
                                                               tmpdf=None, return_df=False)
        self.save_check_point()
        
        # Creation of rolling_values features
        for tick in [['10min', '30min'], ['5min', '30min'], ['15min', '60min']]:
            for col in ['close', 'high', 'low']:
                rolling_values_args = {'column_name': col, 'aggs': ['mean', 'max'],
                                       'last_ticks': tick, 'oper': ['-', '=']}
                self.feature_mart.rolling_values(rolling_values_args, 
                                                 tmpdf=None, return_df=False)
        self.save_check_point()
        
        # Creation of price_data_range_hour features
        for rng in ['price_range', 'price_deviation_max_first_col']:
            for hr in [['09:00', '10:30'], ['10:30', '11:30']]:
                price_data_range_hour_args = {'first_col': 'high', 'second_col': 'low', 
                                              'hour_range': hr, 'range_type': rng}
                self.feature_mart.price_data_range_hour(price_data_range_hour_args, 
                                                        tmpdf=None, return_df=False)
        self.save_check_point()
        
        # Creation of price_velocity_args features
        #for sft in [10, 20, 30, 40, 50, 60]:
        #    price_velocity_args = {'freq': None,
        #                            'shift': sft, 'shift_column': 'close'}
        #    self.feature_mart.price_velocity(price_velocity_args, tmpdf=None, return_df=False)
        #self.save_check_point()
        
        #for frq in ['D']:
        #    for sft in [5, 6, 7]:
        #        price_velocity_args = {'freq': frq,
        #                               'shift': sft, 
        #                               'shift_column': 'close'}
        #        self.feature_mart.price_velocity(price_velocity_args, tmpdf=None, return_df=False)
        #self.save_check_point()

        # Creation of price_velocity_args features
        for col in ['close', 'high', 'low']:
            for wndw in [15, 45, 90, 150,300,500]:
                rolling_zscore_args = {'column_name': col, 'window': wndw}
                self.feature_mart.rolling_zscore(
                    rolling_zscore_args, tmpdf=None, return_df=False)
        self.save_check_point()

        # Creation of rolling_log_transform_args features
        for col in ['close', 'high', 'low']:
            rolling_log_transform_args = {'column_name': col}
            self.feature_mart.rolling_log_transform(
                rolling_log_transform_args, tmpdf=None, return_df=False)
        self.save_check_point()
        
        # Creation of rolling_percentage_change_multiplier features
        for col in ['close', 'high', 'low']:
            for per in [15, 45, 90, 120, 240,500]:
                rolling_percentage_change_multiplier_args = {'column_name': col, 'periods': per, 
                                                             'fill_method': 'pad', 'limit': None, 
                                                             'freq': None, 'multiplier': 100}
                self.feature_mart.rolling_percentage_change_multiplier(rolling_percentage_change_multiplier_args, 
                                                                       tmpdf=None, return_df=False)
        self.save_check_point()
        
        
        # Creation of rolling_weighted_exponential_average features
        for col in ['close','high', 'low']:
            for spn in [22, 66, 110, 120,240,500]:
                rolling_weighted_exponential_average_args = {'column_name': col, 'com': None, 
                                                             'span': spn, 'halflife': None,
                                                             'alpha': None, 'min_periods': 0, 
                                                             'adjust': True, 'ignore_na': False, 
                                                             'axis': 0, 'times': None}
                
                self.feature_mart.rolling_weighted_exponential_average(rolling_weighted_exponential_average_args, 
                                                                       tmpdf=None, return_df=False)
        self.save_check_point()
        
        # Creation of rolling_percentile features
        for col in ['close', 'high', 'low']:
            for window in [15, 45, 60, 90, 120,240,500]:
                for quant in [0.75,0.90]:
                    rolling_percentile_args = {'column_name': col, 'window': window, 
                                               'min_periods': None, 'quantile': quant}
                    self.feature_mart.rolling_percentile(
                        rolling_percentile_args, tmpdf=None, return_df=False)
        self.save_check_point()
        
        # Creation of rolling_rank features
        for col in ['close', 'high', 'low']:
            for window in [15, 45, 60, 90, 120,240,500]:
                rolling_rank_args = {'column_name': col,
                                     'window': window, 'min_periods': None}
                self.feature_mart.rolling_rank(rolling_rank_args, tmpdf=None, return_df=False)
        self.save_check_point()
        
        # Creation of rolling_binning features
        for col in ['close', 'high', 'low']:
            for window in [15, 45, 60, 90, 120,240,500]:
                for n_bin in [3, 5]:
                    rolling_binning_args = {'column_name': col, 'window': window,
                                            'min_periods': None, 'get_current_row_bin': True, 
                                            'n_bins': n_bin}
                    self.feature_mart.rolling_binning(rolling_binning_args, tmpdf=None, return_df=False)
        self.save_check_point()
        
        # Creation of rolling_trends features
        for col in ['close', 'high', 'low']:
            for window in [15, 45, 60, 90, 120,240,500]:
                rolling_trends_args = {'column_name': col,
                                       'window': window, 'min_periods': None}
                self.feature_mart.rolling_trends(rolling_trends_args, tmpdf=None, return_df=False)
        self.save_check_point()

        # Creation of rolling_stats features
        for col in ['close', 'high', 'low']:
            for window in [15, 45, 60, 90, 120,240,500]:
                rolling_stats_args = {'column_name': col,'window': window, 
                                      'min_periods': None}
                self.feature_mart.rolling_stats(rolling_stats_args, tmpdf=None, return_df=False)
        self.save_check_point()

        # Creation of rolling_stats_lookback features
        for col in ['close', 'high', 'low']:
            for window in [15, 45, 60, 90, 120,240,500]:
                rolling_stats_lookback_args = {'column_name': col, 'window': window, 
                                               'min_periods': None, 'lookback_divider': 2}
                self.feature_mart.rolling_stats_lookback(rolling_stats_lookback_args, tmpdf=None, return_df=False)
        self.save_check_point()
        
        # Creation of rolling_stats_lookback_compare features
        for col in ['close', 'high', 'low']:
            for window in [15, 45, 60, 90, 120,240,500]:
                rolling_stats_lookback_compare_args = {'column_name': col, 'window': window, 
                                                       'min_periods': None, 'lookback_divider': 2}
                self.feature_mart.rolling_stats_lookback_compare(
                    rolling_stats_lookback_compare_args, tmpdf=None, return_df=False)
        self.save_check_point()
        
        # Creation of rolling_previous_day_range features
        for col in ['close', 'high', 'low']:
            for resample in ['1min', '15min', '30min']:
                for freq in ['d', 'w']:
                    rolling_previous_day_range_args = {
                        'column_name': col, 'freq': freq, 'shift_val': 1, 'resample': resample}
                    self.feature_mart.rolling_previous_day_range(rolling_previous_day_range_args, 
                                                                 tmpdf=None, return_df=False)
        self.save_check_point()

        # Creation of rolling_gap_open_min features
        for col in ['close', 'high', 'low']:
            rolling_gap_open_min_args = {'column_name': col}
            self.feature_mart.rolling_gap_open_min(rolling_gap_open_min_args, tmpdf=None, return_df=False)
        self.save_check_point()
        
        # Creation of treat_unstable_cols features
        # treat_unstable_cols_args = {'basis_column':'close','ohlc_columns':['close','open','high','low'],'tolerance':17000,'transform_option':'rank'}
        # self.feature_mart.rolling_gap_open_min(
        #        treat_unstable_cols_args, tmpdf=None, return_df=False)
        
        rolling_stats_lookback_args = {'column_name': 'open', 'window': 60, 
                                               'min_periods': None, 'lookback_divider': 2}
        self.feature_mart.rolling_stats_lookback(rolling_stats_lookback_args, tmpdf=None, return_df=False)
        self.save_check_point()
        
        # Creation of timerseries splits features
        if self.time_splitter:
            timeseries_args = {'split_type': 'month','buffer_type':'month','buffer':2,'splits':{'train':[4,8],'validation':[1,3],'test':[1,3]},'split_col_name':'time_split'}        
            self.feature_mart.time_series_column_creator(timeseries_args,df=None, return_df=False)
            self.save_check_point()
        
        # Creation of treat_unstable_cols features
        
        if self.update_unstable:
            self.feature_mart.update_unstable_columns(tolerance=2,filter_ind='time_split_train_fold_10')
            self.save_check_point()
        if self.column_unstable:
            treat_unstable_cols_args = {'basis_column':'close',
                                        'transform_options':['rolling_percentage_change_multiplier','rank']}
            self.feature_mart.treat_unstable_cols(treat_unstable_cols_args)
            self.save_check_point()
        
        self.feature_mart.update_nulls(filter_ind='time_split_train_fold_10',tolerance_count=6000)
        self.save_check_point(end_conection=True)