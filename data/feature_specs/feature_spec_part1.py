
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


class pipelines:
    def __init__(self, master_config_path, master_config_name, db_conection, train_feature_table=None,
                 train_feature_info_table=None,verbose=True):
        self.feature_mart = de.feature_mart(master_config_path=master_config_path,
                                            master_config_name=master_config_name,
                                            db_conection=db_conection,
                                            train_feature_table=train_feature_table,
                                            train_feature_info_table=train_feature_info_table,
                                            verbose=verbose)

    def pipeline_definitions(self):
        
        
        # Creation of labels
        for label in ['label_generator_4class', 'label_generator_7class', 'label_generator_9class', 'label_generator_14class']:
            label_creator_args = {'freq': '1min', 'shift': -15,'shift_column': 'close', 'generator_function_name': label}
            self.feature_mart.label_creator(
                label_creator_args, tmpdf=None, return_df=False)
        '''
        
        # Creation of technical indicators using pandata
        create_technical_indicator_using_pandasta_args = {'exclude': ["jma",
                                                                      "pvo", "vwap", "vwma", "ad", 
                                                                      "adosc", "aobv", "cmf", "efi",
                                                                      "eom", "kvo", "mfi", "nvi", "obv",
                                                                      "pvi", "pvol", "pvr", "pvt"]}
        self.feature_mart.create_technical_indicator_using_pandasta(create_technical_indicator_using_pandasta_args, 
                                                                    tmpdf=None, return_df=False)

        # Creation of technical indicators using signals
        create_technical_indicator_using_signals_args = {'method_type': ['volumn_', 'volatile_', 'transform_', 
                                                                         'cycle_', 'pattern_', 'stats_', 'math_', 'overlap_']}
        self.feature_mart.create_technical_indicator_using_signals(create_technical_indicator_using_signals_args, 
                                                                   tmpdf=None, return_df=False)
        
        # Creation of technical indicators using ta
        create_technical_indicator_using_ta_args = {'open': 'open', 'high': 'high', 'low': 'low', 
                                                    'close': 'close','vectorized': True,
                                                    'fillna': False, 'colprefix': 'ta', 
                                                    'volume_ta': False, 'trend_ta': True, 
                                                    'momentum_ta': True, 'others_ta': True, 'verbose': True}
        self.feature_mart.create_technical_indicator_using_ta(create_technical_indicator_using_ta_args, 
                                                              tmpdf=None, return_df=False)
        
        # Creation of last_tick_greater_values_count features
        for tick in [10, 15, 30, 60, 45, 90, 120]:
            for col in ['close', 'open', 'high', 'low']:
                last_tick_greater_values_count_args = {'column': col, 'last_ticks': tick}
                self.feature_mart.last_tick_greater_values_count(
                    last_tick_greater_values_count_args, tmpdf=None, return_df=False)

        # Creation of price_last_tick_breach_count features
        for tick in [10, 15, 30, 60, 45, 90, 120]:
            for col in ['close', 'open', 'high', 'low']:
                price_last_tick_breach_count_args = {'column_name': col, 'breach_type': 'morethan', 'last_ticks': tick}
                self.feature_mart.price_last_tick_breach_count(price_last_tick_breach_count_args, 
                                                               tmpdf=None, return_df=False)
        
        # Creation of rolling_values features
        for tick in [['10min', '30min'], ['5min', '30min'], ['15min', '60min']]:
            for col in ['close', 'open', 'high', 'low']:
                rolling_values_args = {'column_name': col, 'aggs': ['mean', 'max'],
                                       'last_ticks': tick, 'oper': ['-', '=']}
                self.feature_mart.rolling_values(rolling_values_args, 
                                                 tmpdf=None, return_df=False)
        
        # Creation of price_data_range_hour features
        for rng in ['price_range', 'price_deviation_max_first_col']:
            for hr in [['09:00', '10:30'], ['10:30', '11:30']]:
                price_data_range_hour_args = {'first_col': 'high', 'second_col': 'low', 
                                              'hour_range': hr, 'range_type': rng}
                self.feature_mart.price_data_range_hour(price_data_range_hour_args, 
                                                        tmpdf=None, return_df=False)
        
        # Creation of price_velocity_args features
        for sft in [10, 20, 30, 40, 50, 60]:
            price_velocity_args = {'freq': None,
                                    'shift': sft, 'shift_column': 'close'}
            self.feature_mart.price_velocity(price_velocity_args, tmpdf=None, return_df=False)
        
        for frq in ['D']:
            for sft in [5, 6, 7]:
                price_velocity_args = {'freq': frq,
                                       'shift': sft, 
                                       'shift_column': 'close'}
                self.feature_mart.price_velocity(price_velocity_args, tmpdf=None, return_df=False)

        # Creation of price_velocity_args features
        for col in ['close', 'open', 'high', 'low']:
            for wndw in [15, 30, 45, 60, 90]:
                rolling_zscore_args = {'column_name': col, 'window': wndw}
                self.feature_mart.rolling_zscore(
                    rolling_zscore_args, tmpdf=None, return_df=False)

        # Creation of rolling_log_transform_args features
        for col in ['close', 'open', 'high', 'low']:
            rolling_log_transform_args = {'column_name': col}
            self.feature_mart.rolling_log_transform(
                rolling_log_transform_args, tmpdf=None, return_df=False)
        
        # Creation of rolling_percentage_change_multiplier features
        for col in ['close', 'open', 'high', 'low']:
            for per in [15, 30, 45, 60, 90, 120]:
                rolling_percentage_change_multiplier_args = {'column_name': col, 'periods': per, 
                                                             'fill_method': 'pad', 'limit': None, 
                                                             'freq': None, 'multiplier': 100}
                self.feature_mart.rolling_percentage_change_multiplier(rolling_percentage_change_multiplier_args, 
                                                                       tmpdf=None, return_df=False)
        '''
        '''
        # Creation of rolling_weighted_exponential_average features
        for col in ['close', 'open', 'high', 'low']:
            for spn in [22, 44, 66, 88, 110, 120]:
                rolling_weighted_exponential_average_args = {'column_name': col, 'com': None, 
                                                             'span': spn, 'halflife': None,
                                                             'alpha': None, 'min_periods': 0, 
                                                             'adjust': True, 'ignore_na': False, 
                                                             'axis': 0, 'times': None}
                
                self.feature_mart.rolling_weighted_exponential_average(rolling_weighted_exponential_average_args, 
                                                                       tmpdf=None, return_df=False)
        
        # Creation of rolling_percentile features
        for col in ['close', 'open', 'high', 'low']:
            for window in [10, 15, 30, 45, 60, 90, 120]:
                for quant in [0.75,0.90]:
                    rolling_percentile_args = {'column_name': col, 'window': window, 
                                               'min_periods': None, 'quantile': quant}
                    self.feature_mart.rolling_percentile(
                        rolling_percentile_args, tmpdf=None, return_df=False)
        
        # Creation of rolling_rank features
        for col in ['close', 'open', 'high', 'low']:
            for window in [10, 15, 30, 45, 60, 90, 120]:
                rolling_rank_args = {'column_name': col,
                                     'window': window, 'min_periods': None}
                self.feature_mart.rolling_rank(rolling_rank_args, tmpdf=None, return_df=False)
        
        # Creation of rolling_binning features
        for col in ['close', 'open', 'high', 'low']:
            for window in [10, 15, 30, 45, 60, 90, 120]:
                for n_bin in [3, 5]:
                    rolling_binning_args = {'column_name': col, 'window': window,
                                            'min_periods': None, 'get_current_row_bin': True, 
                                            'n_bins': n_bin}
                    self.feature_mart.rolling_binning(rolling_binning_args, tmpdf=None, return_df=False)
        
        # Creation of rolling_trends features
        for col in ['close', 'open', 'high', 'low']:
            for window in [10, 15, 30, 45, 60, 90, 120]:
                rolling_trends_args = {'column_name': col,
                                       'window': window, 'min_periods': None}
                self.feature_mart.rolling_trends(rolling_trends_args, tmpdf=None, return_df=False)

        # Creation of rolling_stats features
        for col in ['close', 'open', 'high', 'low']:
            for window in [10, 15, 30, 45, 60, 90, 120]:
                rolling_stats_args = {'column_name': col,'window': window, 
                                      'min_periods': None}
                self.feature_mart.rolling_stats(rolling_stats_args, tmpdf=None, return_df=False)

        # Creation of rolling_stats_lookback features
        for col in ['close', 'open', 'high', 'low']:
            for window in [10, 15, 30, 45, 60, 90, 120]:
                rolling_stats_lookback_args = {'column_name': col, 'window': window, 
                                               'min_periods': None, 'lookback_divider': 2}
                self.feature_mart.rolling_stats_lookback(rolling_stats_lookback_args, tmpdf=None, return_df=False)
        
        # Creation of rolling_stats_lookback_compare features
        for col in ['close', 'open', 'high', 'low']:
            for window in [10, 15, 30, 45, 60, 90, 120]:
                rolling_stats_lookback_compare_args = {'column_name': col, 'window': window, 
                                                       'min_periods': None, 'lookback_divider': 2}
                self.feature_mart.rolling_stats_lookback_compare(
                    rolling_stats_lookback_compare_args, tmpdf=None, return_df=False)
        
        # Creation of rolling_previous_day_range features
        for col in ['close', 'open', 'high', 'low']:
            for resample in ['1min', '15min', '30min']:
                for freq in ['d', 'w']:
                    rolling_previous_day_range_args = {
                        'column_name': col, 'freq': freq, 'shift_val': 1, 'resample': resample}
                    self.feature_mart.rolling_previous_day_range(rolling_previous_day_range_args, 
                                                                 tmpdf=None, return_df=False)

        # Creation of rolling_gap_open_min features
        for col in ['close', 'open', 'high', 'low']:
            rolling_gap_open_min_args = {'column_name': col}
            self.feature_mart.rolling_gap_open_min(rolling_gap_open_min_args, tmpdf=None, return_df=False)
        
        # Creation of treat_unstable_cols features
        # treat_unstable_cols_args = {'basis_column':'close','ohlc_columns':['close','open','high','low'],'tolerance':17000,'transform_option':'rank'}
        # self.feature_mart.rolling_gap_open_min(
        #        treat_unstable_cols_args, tmpdf=None, return_df=False)
        
        # Creation of timerseries splits features
        timeseries_args = {'split_type': 'month','buffer_type':'month','buffer':2,'splits':{'train':[4,8],'validation':[1,3],'test':[1,3]},'split_col_name':'time_split'}        
        self.feature_mart.time_series_column_creator(timeseries_args,df=None, return_df=False)
        '''