from config.common.config import Config, DefineConfig
from re import A
from statistics import mean
from tabnanny import verbose
# from signal import Signal
from datetime import timedelta
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import datetime as dt
from talib.abstract import *
import zipfile
import fnmatch
import os
import pandas as pd
import pickle
from pathlib import Path
from feature_engine.discretisation import EqualWidthDiscretiser
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer, ArbitraryNumberImputer, EndTailImputer, DropMissingData
from data.features.signals import Signals, add_all_ta_features
import logging
import data.utils.duckdb_utils as ddu
import data.utils.data_utils as du
import gc
from pandas.api.types import is_numeric_dtype, is_bool_dtype,   is_datetime64_any_dtype, is_string_dtype, is_datetime64_dtype
import random
from dateutil.relativedelta import relativedelta
import duckdb

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def print_log(log, using_print=True):
    if using_print:
        print(log)
    else:
        logging.info(log)

def convert_df_to_timeseries(df):
    df['date_time'] = df['date'].astype(str) + ' ' + df['time']
    df = df.sort_values(by='date_time')
    df.index = df['date_time']
    df = df[['open','high','low','close']]
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    return df

def convert_todate_deduplicate(df):
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df

class LabelCreator(BaseEstimator, TransformerMixin):
    def __init__(self, freq='1min',shift=-15,shift_column='close'):
        self.freq = freq
        self.shift = shift
        self.shift_column = shift_column
        self.label_name = f'label_{shift}_{freq}_{shift_column}'
        
    def fit(self, X, y=None):
        return self    # Nothing to do in fit in this scenario
    
    def label_generator_v2(self,val):
        if val <= 10 and val>=-10:
            return '-10to10'
        elif val > 10 and val <= 20:
            return '10to20'
        elif val > 20 and val <= 40:
            return '20to40'
        elif val > 40 and val <= 60:
            return '40to60'
        elif val > 60 and val <= 80:
            return '60to80'
        elif val > 80 and val <= 100:
            return '80to100'
        elif val > 100:
            return 'above100'
        elif val < -10 and val >= -20:
            return '-10to-20'
        elif val < -20 and val >= -40:
            return '-20to-40'
        elif val < -40 and val >= -60:
            return '-40to-60'
        elif val < -60 and val >= -80:
            return '-60to-80'
        elif val < -80 and val >= -100:
            return '-80to-100'
        elif val < -100:
            return 'below100'
        else:
            return 'unknown'

    def label_generator(self,val):
        if val <= 35 and val>=0:
            return '-0to35'
        elif val > 35 and val <= 80:
            return '35to80'
        elif val > 80 and val <= 150:
            return '80to150'
        elif val > 150:
            return 'above150'
        elif val > -35 and val <= 0:
            return '0to-35'
        elif val > -80 and val <= -35:
            return '-35to-80'
        elif val > -150 and val <= -80:
            return '-80to-150'
        elif val < -150:
            return 'below150'
        else:
            return 'unknown'

    def transform(self, df):
        #df.index = pd.to_datetime(df.index)
        df.index = pd.DatetimeIndex(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df[self.label_name] = df.shift(self.shift, freq=self.freq)[self.shift_column].subtract(df[self.shift_column]).apply(self.label_generator)  
        print_log(f"Shape of dataframe after transform is {df.shape}") 
        return df

class LabelCreator_Light(BaseEstimator, TransformerMixin):
    def __init__(self, freq='1min',shift=-15,shift_column='close'):
        self.freq = freq
        self.shift = shift
        self.shift_column = shift_column
        self.label_name = f'label_{shift}_{freq}_{shift_column}'
        
    def fit(self, X, y=None):
        return self    # Nothing to do in fit in this scenario

    def label_generator(self,val):
        if val <= 30 and val>=0:
            return '-0to30'
        elif val > 30 and val <= 80:
            return '30to80'
        elif val > 80:
            return 'above80'
        elif val > -30 and val <= 0:
            return '0to-30'
        elif val > -80 and val <= -30:
            return '-30to-80'
        elif val <= -80:
            return 'below80'
        else:
            return 'unknown'

    def transform(self, df):
        #df.index = pd.to_datetime(df.index)
        df.index = pd.DatetimeIndex(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df[self.label_name] = df.shift(self.shift, freq=self.freq)[self.shift_column].subtract(df[self.shift_column]).apply(self.label_generator)  
        print_log(f"Shape of dataframe after transform is {df.shape}") 
        return df

class LabelCreator_Super_Light(BaseEstimator, TransformerMixin):
    def __init__(self, freq='1min',shift=-15,shift_column='close'):
        self.freq = freq
        self.shift = shift
        self.shift_column = shift_column
        self.label_name = f'label_{shift}_{freq}_{shift_column}_SL'
        
    def fit(self, X, y=None):
        return self    # Nothing to do in fit in this scenario

    def label_generator(self,val):
        if val <= 30 and val>=-30:
            return 'neutral'
        elif val > 30:
            return 'call'
        elif val < -30:
            return 'put'
        else:
            return 'unknown'

    def transform(self, df):
        #df.index = pd.to_datetime(df.index)
        df.index = pd.DatetimeIndex(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df[self.label_name] = df.shift(self.shift, freq=self.freq)[self.shift_column].subtract(df[self.shift_column]).apply(self.label_generator)  
        print_log(f"Shape of dataframe after transform is {df.shape}") 
        return df
class TechnicalIndicator(BaseEstimator, TransformerMixin):
    def __init__(self,method_type = ['volumn_','volatile_','transform_','cycle_','pattern_','stats_','math_','overlap_']):
        self.method_type = method_type

    def fit(self, X, y=None):
        self.all_methods = []
        a = dict(Signals.__dict__)
        for a1,a2 in a.items():
            self.all_methods.append(a1)
        self.all_methods = [m1 for m1,m2 in a.items() if m1[:1]!='_']
        self.all_methods = [m for m in self.all_methods for mt in self.method_type if mt in m]
        return self    # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        sig = Signals(df)
        self.methods_run = []
        self.methods_notrun = []
        for f in self.all_methods:
            try:
                exec(f'sig.{f}()')
                self.methods_run.append(f)
            except Exception as e1:
                print_log(f"Function {f} was unable to run, Error is {e1}")
                self.methods_notrun.append(f)
        print_log(f"Shape of dataframe after TechnicalIndicator is {df.shape}")
        return sig.df

class CreateTechnicalIndicatorUsingPandasTA(BaseEstimator, TransformerMixin):
    def __init__(self,exclude=["pvo","vwap","vwma","ad","adosc","aobv","cmf","efi","eom","kvo","mfi","nvi","obv","pvi","pvol","pvr","pvt"]
,verbose=True):
        import pandas_ta as ta
        self.exclude = exclude
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        import pandas_ta as ta
        if self.verbose:
            print_log(f"Shape of dataframe before CreateTechnicalIndicatorUsingPandasTA is {df.shape}")
        df.ta.strategy(exclude=self.exclude,verbose=self.verbose,timed=True)
        if self.verbose:
            print_log(f"Shape of dataframe after CreateTechnicalIndicatorUsingPandasTA is {df.shape}") 
        return df

class CreateTechnicalIndicatorUsingTA(BaseEstimator, TransformerMixin):
    def __init__(self, open='open',high='high',low='low',close='close',volume='volume',vectorized=True,fillna=False,colprefix='ta',volume_ta=True,volatility_ta=True,trend_ta=True,momentum_ta=True,others_ta=True,verbose=True):
        self.open=open
        self.high=high
        self.low=low
        self.close=close
        self.volume=volume
        self.fillna=fillna
        self.colprefix=colprefix
        self.volume_ta=volume_ta
        self.volatility_ta=volatility_ta
        self.trend_ta=trend_ta
        self.momentum_ta=momentum_ta
        self.others_ta=others_ta
        self.verbose = verbose
        self.vectorized = vectorized
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        if self.verbose:
            print_log(f"Shape of dataframe before CreateTechnicalIndicatorUsingTA is {df.shape}")
        df = add_all_ta_features(
            df,
            open = self.open,
            high = self.high,
            low = self.low,
            close = self.close,
            volume = self.volume,
            fillna = self.fillna,
            colprefix = self.colprefix,
            vectorized = self.vectorized,
            volume_ta = self.volume_ta,
            volatility_ta  = self.volatility_ta,
            trend_ta  = self.trend_ta,
            momentum_ta  = self.momentum_ta,
            others_ta = self.others_ta,
        )
        if self.verbose:
            print_log(f"Shape of dataframe after CreateTechnicalIndicatorUsingTA is {df.shape}") 
        return df
class NormalizeDataset(BaseEstimator, TransformerMixin):
    def __init__(self, column_pattern = [],columns = [],impute_values=False,impute_type = 'categorical',convert_to_floats = False,arbitrary_impute_variable=99,drop_na_col=False,drop_na_rows=False,
    fillna = False,fillna_method = 'bfill',fill_index=False):
        self.impute_values = impute_values
        self.convert_to_floats = convert_to_floats
        self.impute_type = impute_type
        self.arbitrary_impute_variable = arbitrary_impute_variable
        self.drop_na_col = drop_na_col
        self.drop_na_rows = drop_na_rows
        self.fillna_method = fillna_method
        self.fillna = fillna
        self.column_pattern = column_pattern
        self.columns = columns
        self.fill_index = fill_index

    def fit(self, df, y=None):
        if len(self.columns) == 0:
            self.columns = [m for m in df.columns.tolist() for mt in self.column_pattern if mt in m]
            self.columns = list(set(self.columns))
        return self    # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        print_log(f"Shape of dataframe before NormalizeDataset is {df.shape}")
        info_list = []
        df = convert_todate_deduplicate(df)
        if self.convert_to_floats:
            for col in self.columns:
                df[col] = df[col].astype('float')
                info_list.append('convert_to_floats')
        if self.fill_index:
            df = df.reindex(pd.date_range(min(df.index), max(df.index), freq ='1min'))
            df = df.resample('1min').ffill()
        if self.impute_values:
            from sklearn.pipeline import Pipeline
            if self.impute_type == 'mean_median_imputer':
                imputer = MeanMedianImputer(imputation_method='median', variables=self.columns)
                info_list.append('mean_median_imputer')
            elif self.impute_type == 'categorical':
                imputer = CategoricalImputer(variables=self.columns)
                info_list.append('categorical')
            elif self.impute_type == 'arbitrary':
                if isinstance(self.arbitrary_impute_variable, dict):
                    imputer = ArbitraryNumberImputer(imputer_dict = self.arbitrary_impute_variable)
                    
                else:
                    imputer = ArbitraryNumberImputer(variables = self.columns,arbitrary_number = self.arbitrary_number)
                info_list.append('arbitrary')
            else:
                imputer = CategoricalImputer(variables=self.columns)
                info_list.append('categorical')
            imputer.fit(df)
            df= imputer.transform(df)
        if self.fillna:
            df = df.fillna(method=self.fillna_method)
            info_list.append('fillna')
        if self.drop_na_col:
            imputer = DropMissingData(missing_only=True)
            imputer.fit(df)
            df= imputer.transform(df)
            info_list.append('drop_na_col')
        if self.drop_na_rows:
            df = df.dropna(axis=0)
            info_list.append('drop_na_rows')
        df = df.sort_index()
        print_log(f"Shape of dataframe after NormalizeDataset is {df.shape} : {'.'.join(info_list)}")
        return df
class LastTicksGreaterValuesCount(BaseEstimator, TransformerMixin):
    def __init__(self, column_pattern=[],columns=[],create_new_col = True,last_ticks=10):
        self.columns = columns
        self.last_ticks = last_ticks
        self.create_new_col = create_new_col
        self.column_pattern = column_pattern
        
    def fit(self, df, y=None):
        if len(self.columns) == 0:
            self.columns = [m for m in df.columns.tolist() for mt in self.column_pattern if mt in m]
            self.columns = list(set(self.columns))
        return self   # Nothing to do in fit in this scenario
    
    def rolling_window(self,a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def transform(self, df):
        print_log('*'*100)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]         
        for col in self.columns:
            print_log(f"LastTicksGreaterValuesCount : {col} : f'last_tick_{col}_{self.last_ticks}'")
            x = np.concatenate([[np.nan] * (self.last_ticks), df[col].values])
            arr = self.rolling_window(x, self.last_ticks + 1)
            if self.create_new_col:
                #df[f'last_tick_{col}_{self.last_ticks}'] = self.rolling_window(x, self.#last_ticks + 1)
                df[f'last_tick_{col}_{self.last_ticks}']  = (arr[:, :-1] > arr[:, [-1]]).sum(axis=1)
            else:
                df[col] = (arr[:, :-1] > arr[:, [-1]]).sum(axis=1)
        print_log(f"Shape of dataframe after LastTicksGreaterValuesCount is {df.shape}")
        return df

def convert_todate_deduplicate(df):
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')] 
    return df

class PriceLastTickBreachCountv1(BaseEstimator, TransformerMixin):
    def __init__(self, column_pattern=[],columns=[],create_new_col = True,last_ticks='10min',breach_type = ['mean']):
        self.columns = columns
        self.last_ticks = last_ticks
        self.create_new_col = create_new_col
        self.breach_type = breach_type
        self.column_pattern = column_pattern
        
    def fit(self, df, y=None):
        if len(self.columns) == 0:
            self.columns = [m for m in df.columns.tolist() for mt in self.column_pattern if mt in m]
            self.columns = list(set(self.columns))
        return self    # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()        
        for col in self.columns:
            for breach_type in self.breach_type:
                
                if self.create_new_col:
                    col_name = f'last_tick_{breach_type}_{col}_{self.last_ticks}'
                else:
                    col_name = col
                print_log(f"PriceLastTickBreachCount : {breach_type} : {col} : {col_name}")
                if breach_type == 'morethan':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x[-1] > x[:-1]).sum()).fillna(0)
                elif breach_type == 'lessthan':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x[-1] < x[:-1]).sum()).fillna(0)
                elif breach_type == 'mean':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].mean()).sum()).fillna(0).astype(int)
                elif breach_type == 'min':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].min()).sum()).fillna(0).astype(int)
                elif breach_type == 'max':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].max()).sum()).fillna(0).astype(int)
                elif breach_type == 'median':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].median()).sum()).fillna(0).astype(int)
                elif breach_type == '10thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.1)).sum()).fillna(0).astype(int)
                elif breach_type == '25thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.25)).sum()).fillna(0).astype(int)
                elif breach_type == '75thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.75)).sum()).fillna(0).astype(int)
                elif breach_type == '95thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.95)).sum()).fillna(0).astype(int)
                else:
                    df[col_name] = (df[col].rolling(self.last_ticks, min_periods=1)
                            .apply(lambda x: (x[-1] > x[:-1]).mean())
                            .astype(int))
        print_log(f"Shape of dataframe after PriceLastTickBreachCount is {df.shape}")
        return df

class ValueLastTickBreachCount(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[],column_pattern=[],create_new_col = True,last_ticks='5',breach_type = ['morethan'],verbose=False):
        self.columns = columns
        self.last_ticks = last_ticks
        self.create_new_col = create_new_col
        self.breach_type = breach_type
        self.self.verbose = self.verbose
        self.column_pattern = column_pattern
        
    def fit(self, df, y=None):
        if len(self.columns) == 0:
            self.columns = [m for m in df.columns.tolist() for mt in self.column_pattern if mt in m]
            self.columns = list(set(self.columns))
        return self    # Nothing to do in fit in this scenario

    def transform(self, df):   
        print_log('*'*100)
        df = df.sort_index()    
        for col in self.columns:
            for breach_type in self.breach_type:
                
                if self.create_new_col:
                    col_name = f'last_tick_{breach_type}_{col}_{self.last_ticks}'
                else:
                    col_name = col
                print_log(f"ValueLastTickBreachCount : {breach_type} : {col} : {col_name}")
                if breach_type == 'morethan':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x[-1] > x[:-1]).sum()).fillna(0)
                elif breach_type == 'lessthan':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x[-1] < x[:-1]).sum()).fillna(0)
                elif breach_type == 'mean':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].mean()).sum()).fillna(0).astype(int)
                elif breach_type == 'min':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].min()).sum()).fillna(0).astype(int)
                elif breach_type == 'max':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].max()).sum()).fillna(0).astype(int)
                elif breach_type == 'median':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].median()).sum()).fillna(0).astype(int)
                elif breach_type == '10thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.1)).sum()).fillna(0).astype(int)
                elif breach_type == '25thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.25)).sum()).fillna(0).astype(int)
                elif breach_type == '75thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.75)).sum()).fillna(0).astype(int)
                elif breach_type == '95thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.95)).sum()).fillna(0).astype(int)
                else:
                    df[col_name] = (df[col].rolling(self.last_ticks, min_periods=1)
                            .apply(lambda x: (x[-1] > x[:-1]).mean())
                            .astype(int))
        if self.self.verbose:
            print_log(f"Shape of dataframe after ValueLastTickBreachCount is {df.shape}")
        return df

class PriceLastTickBreachCount(BaseEstimator, TransformerMixin):
    def __init__(self, column_pattern=[],columns=[],last_ticks='10min',breach_type = ['mean']):
        self.columns = columns
        self.last_ticks = last_ticks
        self.breach_type = breach_type
        self.column_pattern = column_pattern
        
    def fit(self, df, y=None):
        if len(self.columns) == 0:
            self.columns = [m for m in df.columns.tolist() for mt in self.column_pattern if mt in m]
            self.columns = list(set(self.columns))
        return self    # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()   
        for breach_type in self.breach_type:
            print_log(f"PriceLastTickBreachCount : {breach_type} : {self.last_ticks}")
            if breach_type == 'morethan':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x[-1] > np.array(x[:-1]))).fillna(0)
            elif breach_type == 'lessthan':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x[-1] < np.array(x[:-1]))).fillna(0)
            elif breach_type == 'mean':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x > np.mean(np.array(x)))).fillna(0).astype(int)
            elif breach_type == 'min':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x > np.min(np.array(x)))).fillna(0).astype(int)
            elif breach_type == 'max':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x > np.max(np.array(x)))).fillna(0).astype(int)
            elif breach_type == 'median':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x > np.median(np.array(x)))).fillna(0).astype(int)
            elif breach_type == '10thquantile':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x > np.quantile(np.array(x),0.1))).fillna(0).astype(int)
            elif breach_type == '25thquantile':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x > np.quantile(np.array(x),0.25))).fillna(0).astype(int)
            elif breach_type == '75thquantile':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x > np.quantile(np.array(x),0.75))).fillna(0).astype(int)
            elif breach_type == '95thquantile':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x > np.quantile(np.array(x),0.95))).fillna(0).astype(int)
            else:
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x[-1] > np.array(x[:-1]))).fillna(0)
            col_names = [f"{col}_{self.last_ticks}_{'_'.join(self.breach_type)}_last_tick_" for col in self.columns]
            tmpdf.columns = col_names
            df = pd.merge(df, tmpdf, left_index=True, right_index=True,how='left')
        print_log(f"Shape of dataframe after PriceLastTickBreachCount is {df.shape}")
        return df

class RollingValues(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[],column_pattern=[],last_ticks=['5min','10min'],aggs=['mean','max'],oper = ['-','='],verbose=True):
        self.columns = columns
        self.last_ticks = last_ticks
        self.verbose = verbose
        self.column_pattern = column_pattern
        self.aggs = aggs
        self.oper = oper
        
    def fit(self, df, y=None):
        if len(self.columns) == 0:
            self.columns = [m for m in df.columns.tolist() for mt in self.column_pattern if mt in m]
            self.columns = list(set(self.columns))
        return self    # Nothing to do in fit in this scenario

    def transform(self, df):   
        print_log('*'*100)
        df = df.sort_index()
        eval_stmt = '' 
        for lt,oper,agg in zip(self.last_ticks,self.oper,self.aggs):
            tmpst = f"df[{self.columns}].rolling('{lt}', min_periods=1).{agg}() {oper}"
            eval_stmt = eval_stmt + tmpst
        tmpdf = eval(eval_stmt[:-1])
        col_names = [f"{shftcol}_{'_'.join(self.last_ticks)}_{'_'.join(self.aggs)}_rolling_values" for shftcol in self.columns]
        tmpdf.columns = col_names
        df = pd.merge(df, tmpdf, left_index=True, right_index=True,how='left')
        if self.verbose:
            print_log(f"Shape of dataframe after RollingValues is {df.shape}")
        return df

class PriceDayRangeHourWise(BaseEstimator, TransformerMixin):
    def __init__(self, first_col = 'high',second_col='low',hour_range = [['09:00', '10:30'],['10:30', '11:30']],range_type=['price_range','price_deviation_max_first_col']):
        self.hour_range = hour_range
        self.first_col = first_col
        self.second_col = second_col
        self.range_type = range_type
        
    def fit(self, X, y=None):
        return self    

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        for r1,r2 in self.hour_range:
            for rt in self.range_type:
                print_log(f"PriceDayRangeHourWise : {self.first_col} : {self.second_col} : {r1} : {r2} : {rt}")
                if rt == 'price_range':
                    s1 = df[self.first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max() - df[self.second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
                elif rt == 'price_deviation_max_first_col':
                    s1 = df[self.first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - df[self.first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max()
                elif rt == 'price_deviation_min_first_col':
                    s1 = df[self.first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - df[self.first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
                elif rt == 'price_deviation_max_second_col':
                    s1 = df[self.second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - df[self.second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max()
                elif rt == 'price_deviation_min_second_col':
                    s1 = df[self.second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - df[self.second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
                else:
                    s1 = df[self.first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max() - df[self.second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
            s1.index = pd.to_datetime(s1.index) 
            s1 = s1.sort_index()
            c = [int(i) for i in r2.split(':')]
            s1.index = s1.index + pd.DateOffset(minutes=c[0]*60 + c[1])
            col_name = f"PDR_{self.first_col}_{self.second_col}_{rt}_{r1.replace(':','')}_{r2.replace(':','')}"
            s1.name = col_name
            df=pd.merge(df,s1, how='outer', left_index=True, right_index=True)
            df[col_name] = df[col_name].fillna(method='ffill')
        print_log(f"Shape of dataframe after PriceDayRangeHourWise is {df.shape}")
        return df

class PriceVelocityv2(BaseEstimator, TransformerMixin):
    def __init__(self, freq='D',shift=5,shift_column=['close','open'],shift_column_pattern=[],verbose=False):
        self.freq = freq
        self.shift = shift
        self.shift_column = shift_column
        self.verbose = verbose
        self.shift_column_pattern = shift_column_pattern
        
    def fit(self, df, y=None):
        if len(self.shift_column) == 0:
            self.shift_column = [m for m in df.columns.tolist() for mt in self.shift_column_pattern if mt in m]
            self.shift_column = list(set(self.shift_column))
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        for shftcol in self.shift_column:
            
            if self.freq is not None:
                self.col_name = f'price_velocity_{shftcol}_{self.freq}_{self.shift}'
                print_log(f"PriceVelocity : {shftcol} : {self.col_name}")
                a = df.shift(self.shift, freq=self.freq)[shftcol]
                a.name = self.col_name
                df = pd.merge(df, a, left_index=True, right_index=True,how='left')
                df[self.col_name] = df[shftcol] - df[self.col_name]
                df[self.col_name] = df[self.col_name].round(3)
            else:
                self.col_name = f'price_velocity_{shftcol}_{self.shift}'
                print_log(f"PriceVelocity : {shftcol} : {self.col_name}")
                a = df.shift(self.shift)[shftcol]
                a.name = self.col_name
                df = pd.merge(df, a, left_index=True, right_index=True,how='left')
                df[self.col_name] = df[shftcol] - df[self.col_name]
                df[self.col_name] = df[self.col_name].round(3)
        if self.verbose:
            print_log(f"Shape of dataframe after PriceVelocity is {df.shape}") 
        return df

class PriceVelocity(BaseEstimator, TransformerMixin):
    def __init__(self, freq='D',shift=5,shift_column=['close','open'],shift_column_pattern=[],verbose=False):
        self.freq = freq
        self.shift = shift
        self.shift_column = shift_column
        self.verbose = verbose
        self.shift_column_pattern = shift_column_pattern
        
    def fit(self, df, y=None):
        
        if len(self.shift_column) == 0:
            self.shift_column = [m for m in df.columns.tolist() for mt in self.shift_column_pattern if mt in m]
            self.shift_column = list(set(self.shift_column))
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.freq is not None:
            tmpdf = df[self.shift_column].subtract(df.shift(self.shift,freq=self.freq)[self.shift_column])
            col_names = [f'{shftcol}_{self.freq}_{self.shift}_price_velocity' for shftcol in self.shift_column]
            tmpdf.columns = col_names
        else:
            tmpdf = df[self.shift_column].subtract(df.shift(self.shift)[self.shift_column])
            col_names = [f'{shftcol}_{self.shift}_price_velocity' for shftcol in self.shift_column]
            tmpdf.columns = col_names
        df = pd.merge(df, tmpdf, left_index=True, right_index=True,how='left')
        if self.verbose:
            print_log(f"Shape of dataframe after PriceVelocity is {df.shape}") 
        return df

class PriceVelocityv1(BaseEstimator, TransformerMixin):
    def __init__(self, freq='D',shift=5,shift_column=['close','open'],shift_column_pattern=[],verbose=False):
        self.freq = freq
        self.shift = shift
        self.shift_column = shift_column
        self.verbose = verbose
        self.shift_column_pattern = shift_column_pattern
        
    def fit(self, df, y=None):
        if len(self.shift_column) == 0:
            self.shift_column = [m for m in df.columns.tolist() for mt in self.shift_column_pattern if mt in m]
            self.shift_column = list(set(self.shift_column))
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        for shftcol in self.shift_column:
            print_log(f"PriceVelocity : {shftcol}")
            if self.freq is not None:
                self.col_name = f'price_velocity_{shftcol}_{self.freq}_{self.shift}'
                #df[self.col_name] = df[shftcol].subtract(df.shift(self.shift, freq=self.freq)[shftcol])
                df[self.col_name] = df[shftcol] - df.shift(self.shift, freq=self.freq)[shftcol]
                df[self.col_name] = df[self.col_name].round(3)
            else:
                self.col_name = f'price_velocity_{shftcol}_{self.shift}'
                df[self.col_name] = df[shftcol] - df.shift(self.shift)[shftcol]
                df[self.col_name] = df[self.col_name].round(3)
        if self.verbose:
            print_log(f"Shape of dataframe after PriceVelocity is {df.shape}") 
        return df

class PricePerIncrementv1(BaseEstimator, TransformerMixin):
    def __init__(self, freq='D',shift=5,shift_column=['close','open'],shift_column_pattern=[],verbose=False):
        self.freq = freq
        self.shift = shift
        self.shift_column = shift_column
        self.verbose = verbose
        self.shift_column_pattern = shift_column_pattern
        
    def fit(self, df, y=None):
        if len(self.shift_column) == 0:
            self.shift_column = [m for m in df.columns.tolist() for mt in self.shift_column_pattern if mt in m]
            self.shift_column = list(set(self.shift_column))
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        for shftcol in self.shift_column:
            
            if self.freq is not None:
                self.col_name = f'price_pervelocity_{shftcol}_{self.freq}_{self.shift}'
                print_log(f"PricePerIncrement : {shftcol} : {self.col_name}")
                a = df.shift(self.shift, freq=self.freq)[shftcol]
                a.name = self.col_name
                df = pd.merge(df, a, left_index=True, right_index=True,how='left')
                df[self.col_name] = df[shftcol] - df[self.col_name]
                df[self.col_name] = df[self.col_name]/int(self.shift)
                df[self.col_name] = df[self.col_name].round(4)
            else:
                self.col_name = f'price_pervelocity_{shftcol}_{self.shift}'
                print_log(f"PricePerIncrement : {shftcol} : {self.col_name}")
                a = df.shift(self.shift)[shftcol]
                a.name = self.col_name
                df = pd.merge(df, a, left_index=True, right_index=True,how='left')
                df[self.col_name] = df[shftcol] - df[self.col_name]
                df[self.col_name] = df[self.col_name]/int(self.shift)
                df[self.col_name] = df[self.col_name].round(4)
        if self.verbose:
            print_log(f"Shape of dataframe after PriceVelocity is {df.shape}") 
        return df

class PricePerIncrement(BaseEstimator, TransformerMixin):
    def __init__(self, freq='D',shift=5,shift_column=['close','open'],shift_column_pattern=[],verbose=False):
        self.freq = freq
        self.shift = shift
        self.shift_column = shift_column
        self.verbose = verbose
        self.shift_column_pattern = shift_column_pattern
        
    def fit(self, df, y=None):
        if len(self.shift_column) == 0:
            self.shift_column = [m for m in df.columns.tolist() for mt in self.shift_column_pattern if mt in m]
            self.shift_column = list(set(self.shift_column))
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.freq is not None:
            tmpdf = df[self.shift_column].subtract(df.shift(self.shift,freq=self.freq)[self.shift_column])
            
            col_names = [f'{shftcol}_{self.freq}_{self.shift}_price_per_velocity' for shftcol in self.shift_column]
            tmpdf.columns = col_names
        else:
            tmpdf = df[self.shift_column].subtract(df.shift(self.shift)[self.shift_column])
            col_names = [f'{shftcol}_{self.shift}_price_per_velocity' for shftcol in self.shift_column]
            tmpdf.columns = col_names
        tmpdf = tmpdf/self.shift
        df = pd.merge(df, tmpdf, left_index=True, right_index=True,how='left')
        if self.verbose:
            print_log(f"Shape of dataframe after PricePerVelocity is {df.shape}") 
        return df

class FilterData(BaseEstimator, TransformerMixin):
    def __init__(self, start_date=None,end_date=None,filter_rows=None,verbose=False):
        self.start_date = start_date
        self.end_date = end_date
        self.filter_rows = filter_rows
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self      
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        print_log(f"Value of start_date is {self.start_date}") 
        print_log(f"Value of end_date is {self.end_date}") 
        print_log(f"Value of filter_rows is {self.filter_rows}")
        if self.verbose:
            print_log(f"Shape of dataframe before FilterData is {df.shape}") 
        if self.start_date != 'None' and self.end_date == 'None':
            df = df.sort_index().loc[self.start_date:]
            print_log(f"Data filtered with {self.start_date}") 
        elif self.start_date == 'None' and self.end_date != 'None':
            df = df.sort_index().loc[:self.end_date]
            print_log(f"Data filtered with {self.end_date}") 
        elif self.start_date != 'None' and self.end_date != 'None':
            df = df.sort_index().loc[self.start_date:self.end_date]
            print_log(f"Data filtered with {self.end_date}") 
        else:
            df = df.sort_index()
            print_log(f"No filtering done") 
        if self.filter_rows != 'None':
            df = df[:self.filter_rows]
            print_log(f"Data filtered with filter rows {self.filter_rows}") 
        if self.verbose:
            print_log(f"Shape of dataframe after FilterData is {df.shape}") 
        return df

class Zscoring(BaseEstimator, TransformerMixin):
    def __init__(self, columns,window = 30,verbose=False):
        self.columns = columns
        self.verbose = verbose
        self.window = window
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def zscore(self,x, window):
        r = x.rolling(window=window)
        m = r.mean().shift(1)
        s = r.std(ddof=0).shift(1)
        z = (x-m)/s
        return z

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before Zscoring is {df.shape}") 
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'Zscore_{col}_{self.window}':self.zscore(df[col],self.window)})
            print_log(f"Zscore_{col}_{self.window} completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after Zscoring is {df.shape}") 
        return df

class LogTransform(BaseEstimator, TransformerMixin):
    def __init__(self, columns,verbose=False):
        self.columns = columns
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before LogTransform is {df.shape}") 
        for col in self.columns:
            df[f'Log_{col}'] =df[col].apply(np.log)
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'Log_{col}':df[col].apply(np.log)})
            print_log(f"Log_{col} completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after LogTransform is {df.shape}") 
        return df

class PercentageChange(BaseEstimator, TransformerMixin):
    def __init__(self, columns,periods=30, fill_method='pad', limit=None, freq=None,verbose=False):
        self.columns = columns
        self.periods = periods
        self.fill_method = fill_method
        self.limit = limit
        self.freq = freq
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before PercentageChange is {df.shape}")
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'PerChg_{col}_{self.periods}_{self.freq}':df[col].pct_change(periods=self.periods,fill_method=self.fill_method,limit = self.limit,freq=self.freq)})
            #df[f'PerChg_{col}_{self.periods}_{self.freq}'] =df[col].pct_change(periods=self.periods,fill_method=self.fill_method,limit = self.limit,freq=self.freq)
            print_log(f"PerChg_{col}_{self.periods}_{self.freq} completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after PercentageChange is {df.shape}") 
        return df

class PercentageChange_Multiplier(BaseEstimator, TransformerMixin):
    def __init__(self, columns,periods=30, fill_method='pad', limit=None, freq=None,multiplier=100,verbose=False):
        self.columns = columns
        self.periods = periods
        self.fill_method = fill_method
        self.limit = limit
        self.freq = freq
        self.multiplier = multiplier
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before PercentageChange is {df.shape}")
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'PerChg_{col}_{self.periods}_{self.freq}':df[col].pct_change(periods=self.periods,fill_method=self.fill_method,limit = self.limit,freq=self.freq)*self.multiplier})
            #df[f'PerChg_{col}_{self.periods}_{self.freq}'] =df[col].pct_change(periods=self.periods,fill_method=self.fill_method,limit = self.limit,freq=self.freq)
            print_log(f"PerChg_{col}_{self.periods}_{self.freq} completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after PercentageChange is {df.shape}") 
        return df

class WeightedExponentialAverage(BaseEstimator, TransformerMixin):
    def __init__(self, columns,com=None, span=44, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, axis=0, times=None, verbose=False):
        self.columns = columns
        self.com = com
        self.span = span
        self.halflife = halflife
        self.alpha = alpha
        self.min_periods = min_periods
        self.adjust = adjust
        self.ignore_na = ignore_na
        self.axis = axis
        self.times = times
        #self.method = method
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before WeightedExponentialAverage is {df.shape}")
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'WEA_{col}_{self.span}':df[col].ewm(com=self.com, span=self.span, halflife=self.halflife, alpha=self.alpha, min_periods=self.min_periods, adjust=self.adjust, ignore_na=self.ignore_na, axis=self.axis, times=self.times).mean()})
            print_log(f"WEA_{col}_{self.span} completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after WeightedExponentialAverage is {df.shape}") 
        return df

class PercentileTransform(BaseEstimator, TransformerMixin):
    def __init__(self, columns,window=30,min_periods=None,quantile=0.75,verbose=True):
        self.columns = columns
        self.window = window
        self.min_periods = min_periods
        self.quantile = quantile
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before PercentileTransform is {df.shape}")
        #rollrank_fxn = lambda x: x.rolling(self.window, min_periods=self.min_periods).apply(lambda x: pd.Series(x).quantile(0.75))
        #for col in self.columns:
        #    df[f'PCTL_{col}_{self.window}_{self.min_periods}'] =df[col].rolling(self.window, min_periods=self.min_periods).apply(lambda x: pd.Series(x).quantile(self.quantile))
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'PCTL_{col}_{self.window}_{self.min_periods}':df[col].rolling(self.window, min_periods=self.min_periods).apply(lambda x: np.array(x)[-1] - np.quantile(np.array(x),q=self.quantile))})
            print_log(f"PCTL_{col}_{self.window}_{self.min_periods} completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after PercentileTransform is {df.shape}") 
        return df

class RollingRank(BaseEstimator, TransformerMixin):
    def __init__(self, columns,window=30,min_periods=None,verbose=True):
        self.columns = columns
        self.window = window
        self.min_periods = min_periods
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def rank(self,s):
        #s = pd.Series(array)
        return s.rank(ascending=False)[len(s)-1]

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before RollingRank is {df.shape}")
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'RRNK_{col}_{self.window}_{self.min_periods}':df[col].rolling(window=self.window,min_periods=self.min_periods).apply(self.rank)})
            print_log(f"RRNK_{col}_{self.window}_{self.min_periods} completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after RollingRank is {df.shape}") 
        return df

class BinningTransform(BaseEstimator, TransformerMixin):
    def __init__(self, columns,window=30,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True):
        self.columns = columns
        self.get_current_row_bin = get_current_row_bin
        self.n_bins = n_bins
        self.verbose = verbose
        self.window = window
        self.min_period = min_period

    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before BinningTransform is {df.shape}")
        if self.get_current_row_bin:
            bin_roll_fxn = lambda x: pd.qcut(np.array(x),labels=False,q=self.n_bins,duplicates='drop')[-1]
        else:
            bin_roll_fxn = lambda x: pd.qcut(np.array(x),labels=False,q=self.n_bins,duplicates='drop')[0]
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'BINT_{col}_{self.window}_{self.min_period}_{self.n_bins}':df[col].rolling(window=self.window,min_periods=self.min_period).apply(bin_roll_fxn)})
            print_log(f"BINT_{col}_{self.window}_{self.min_period}_{self.n_bins} completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after BinningTransform is {df.shape}") 
        return df

class PositiveNegativeTrends(BaseEstimator, TransformerMixin):
    def __init__(self, columns,window=30,min_periods=None,verbose=True):
        self.columns = columns
        self.window = window
        self.min_periods = min_periods
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before PositiveNegativeTrends is {df.shape}")
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'PNT_{col}_{self.window}_{self.min_periods}_DIFF':df[col].pct_change().apply(np.sign).rolling(self.window, min_periods=self.min_periods).apply(np.sum)})
            print_log(f"PNT_{col}_{self.window}_{self.min_periods}_DIFF completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after PositiveNegativeTrends is {df.shape}") 
        return df

class Rolling_Stats(BaseEstimator, TransformerMixin):
    def __init__(self, columns,window=30,min_periods=None,verbose=True):
        self.columns = columns
        self.window = window
        self.min_periods = min_periods
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before Rolling_Stats is {df.shape}")
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'RR_{col}_{self.window}_{self.min_periods}_DIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lambda x: np.array(x)[-1]-np.array(x)[0])})
            print_log(f"RR_{col}_{self.window}_{self.min_periods}_DIFF completed")
            merge_dict.update({f'RR_{col}_{self.window}_{self.min_periods}_MAXDIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lambda x: np.array(x)[-1]-max(np.array(x)))})
            print_log(f"RR_{col}_{self.window}_{self.min_periods}_MAXDIFF completed")
            merge_dict.update({f'RR_{col}_{self.window}_{self.min_periods}_MINDIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lambda x: np.array(x)[-1]-min(np.array(x)))})
            print_log(f"RR_{col}_{self.window}_{self.min_periods}_MINDIFF completed")
            merge_dict.update({f'RR_{col}_{self.window}_{self.min_periods}_MEANDIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lambda x: np.array(x)[-1]-mean(np.array(x)))})
            print_log(f"RR_{col}_{self.window}_{self.min_periods}_MEANDIFF completed")
            merge_dict.update({f'RR_{col}_{self.window}_{self.min_periods}_MAXMIN':df[col].rolling(self.window, min_periods=self.min_periods).apply(lambda x: max(np.array(x))-min(np.array(x)))})
            print_log(f"RR_{col}_{self.window}_{self.min_periods}_MAXMIN completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after Rolling_Stats is {df.shape}") 
        return df

class Rolling_Stats_withLookBack(BaseEstimator, TransformerMixin):
    def __init__(self, columns,window=60,lookback_divider =2 ,min_periods=None,verbose=True):
        self.columns = columns
        self.window = window
        self.min_periods = min_periods
        self.verbose = verbose
        self.lookback_divider = lookback_divider
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        def lookback_diff(vals):
            offset = len(vals)//self.lookback_divider
            return np.array(vals)[offset]-np.array(vals)[0]

        def lookback_max(vals):
            offset = len(vals)//self.lookback_divider
            return np.array(vals)[offset]-max(np.array(vals)[0:offset+1])

        def lookback_min(vals):
            offset = len(vals)//self.lookback_divider
            return np.array(vals)[offset]-min(np.array(vals)[0:offset+1])

        def lookback_mean(vals):
            offset = len(vals)//self.lookback_divider
            return np.array(vals)[offset]-np.array(vals)[0:offset+1].mean()

        def lookback_max_max(vals):
            offset = len(vals)//self.lookback_divider
            return max(np.array(vals))-np.array(vals)[0:offset+1].mean()

        def lookback_min_min(vals):
            offset = len(vals)//self.lookback_divider
            return min(np.array(vals))-np.array(vals)[0:offset+1].mean()

        if self.verbose:
            print_log(f"Shape of dataframe before Rolling_Stats_withLookBack is {df.shape}")
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'RRLB_{col}_{self.window}_{self.min_periods}_DIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_diff)})
            print_log(f"RRLB_{col}_{self.window}_{self.min_periods}_DIFF completed")
            merge_dict.update({f'RRLB_{col}_{self.window}_{self.min_periods}_MAXDIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_max)})
            print_log(f"RRLB_{col}_{self.window}_{self.min_periods}_MAXDIFF completed")
            merge_dict.update({f'RRLB_{col}_{self.window}_{self.min_periods}_MINDIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_min)})
            print_log(f"RRLB_{col}_{self.window}_{self.min_periods}_MINDIFF completed")
            merge_dict.update({f'RRLB_{col}_{self.window}_{self.min_periods}_MEANDIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_mean)})
            print_log(f"RRLB_{col}_{self.window}_{self.min_periods}_MEANDIFF completed")
            merge_dict.update({f'RRLB_{col}_{self.window}_{self.min_periods}_MAXMAX':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_max_max)})
            print_log(f"RRLB_{col}_{self.window}_{self.min_periods}_MAXMAX completed")
            merge_dict.update({f'RRLB_{col}_{self.window}_{self.min_periods}_MINMIN':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_min_min)})
            print_log(f"RRLB_{col}_{self.window}_{self.min_periods}_MINMIN completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after Rolling_Stats_withLookBack is {df.shape}") 
        return df

class Rolling_Stats_withLookBack_Compare(BaseEstimator, TransformerMixin):
    def __init__(self, columns,window=60,lookback_divider =2 ,min_periods=None,verbose=True):
        self.columns = columns
        self.window = window
        self.min_periods = min_periods
        self.verbose = verbose
        self.lookback_divider = lookback_divider
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        def lookback_diff(vals):
            offset = len(vals)//self.lookback_divider
            res = (np.array(vals)[offset]-np.array(vals)[-1]) - (np.array(vals)[0]-np.array(vals)[offset])
            return res

        def lookback_max(vals):
            offset = len(vals)//self.lookback_divider
            return max(np.array(vals)[offset+1:])-max(np.array(vals)[0:offset+1])

        def lookback_min(vals):
            offset = len(vals)//self.lookback_divider
            return min(np.array(vals)[offset+1:])-min(np.array(vals)[0:offset+1])

        def lookback_mean(vals):
            offset = len(vals)//self.lookback_divider
            return np.array(vals)[offset+1:].mean()-np.array(vals)[0:offset+1].mean()

        def lookback_max_min(vals):
            offset = len(vals)//self.lookback_divider
            return max(np.array(vals)[offset+1:])-min(np.array(vals)[0:offset+1])

        def lookback_min_max(vals):
            offset = len(vals)//self.lookback_divider
            return min(np.array(vals)[offset+1:])-max(np.array(vals)[0:offset+1])

        if self.verbose:
            print_log(f"Shape of dataframe before Rolling_Stats_withLookBack_Compare is {df.shape}")
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'RRLBC_{col}_{self.window}_{self.min_periods}_DIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_diff)})
            print_log(f"RRLBC_{col}_{self.window}_{self.min_periods}_DIFF completed")
            merge_dict.update({f'RRLBC_{col}_{self.window}_{self.min_periods}_MAXDIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_max)})
            print_log(f"RRLBC_{col}_{self.window}_{self.min_periods}_MAXDIFF completed")
            merge_dict.update({f'RRLBC_{col}_{self.window}_{self.min_periods}_MINDIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_min)})
            print_log(f"RRLBC_{col}_{self.window}_{self.min_periods}_MINDIFF completed")
            merge_dict.update({f'RRLBC_{col}_{self.window}_{self.min_periods}_MEANDIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_mean)})
            print_log(f"RRLBC_{col}_{self.window}_{self.min_periods}_MEANDIFF completed")
            merge_dict.update({f'RRLBC_{col}_{self.window}_{self.min_periods}_MAXMIN':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_max_min)})
            print_log(f"RRLBC_{col}_{self.window}_{self.min_periods}_MAXMIN completed")
            merge_dict.update({f'RRLBC_{col}_{self.window}_{self.min_periods}_MINMAX':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_min_max)})
            print_log(f"RRLBC_{col}_{self.window}_{self.min_periods}_MINMAX completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after Rolling_Stats_withLookBack_Compare is {df.shape}") 
        return df

class PreviousDaysRange(BaseEstimator, TransformerMixin):
    def __init__(self, columns,freq='d',shift=1,resample='1min',verbose=True):
        '''
        freq should always be d or more than d. Any discontinuity in data would cause problem
        '''
        self.columns = columns
        self.freq = freq
        self.shift = shift
        self.resample = resample
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before PreviousDaysRange is {df.shape}")
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'PVDR_{col}_{self.freq}':df[col].resample(self.resample).ffill().groupby(pd.Grouper(freq=self.freq)).apply(lambda x: np.array(x)[-1]-np.array(x)[0]).shift(self.shift)})
            merge_dict.update({f'PVDR_{col}_{self.freq}_MAX_MIN':df[col].resample(self.resample).ffill().groupby(pd.Grouper(freq=self.freq)).apply(lambda x: max(np.array(x))-min(np.array(x))).shift(self.shift)})
            print_log(f"PVDR_{col}_{self.freq}_{self.shift} completed")
        df = pd.merge_asof(df, pd.concat(merge_dict,axis=1), left_index=True, right_index=True)
        if self.verbose:
            print_log(f"Shape of dataframe after PreviousDaysRange is {df.shape}") 
        return df

class GapOpenMinuteChart(BaseEstimator, TransformerMixin):
    def __init__(self, columns,verbose=True):
        self.columns = columns
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before GapOpenMinuteChart is {df.shape}")
        merge_dict = {}
        for col in self.columns:
            tmp = df[col].resample('d').bfill().groupby(pd.Grouper(freq='d')).apply(lambda x:x[0]).subtract( df[col].resample('d').ffill().groupby(pd.Grouper(freq='d')).apply(lambda x:x[-1]).fillna(0))
            merge_dict.update({'GOMC_{col}':tmp[1:]})
            print_log(f"GOMC_{col} completed")
        df = pd.merge_asof(df, pd.concat(merge_dict,axis=1), left_index=True, right_index=True)
        if self.verbose:
            print_log(f"Shape of dataframe after GapOpenMinuteChart is {df.shape}") 
        return df
class ConvertUnstableCols(BaseEstimator, TransformerMixin):
    def __init__(self,basis_column='close',ohlc_columns = ['close','open','high','low'],tolerance=17000,transform_option='rank',verbose=True):
        self.basis_column = basis_column
        self.tolerance = tolerance
        self.transform_option = transform_option
        self.ohlc_columns = ohlc_columns
        self.verbose = verbose
        
    def fit(self, df, y=None):
        self.unstable_cols = [col for col in [c for c in df.columns.tolist() if df[c].dtypes != 'object'] if np.mean(df[col]) > self.tolerance ]
        self.unstable_cols = [col for col in self.unstable_cols if col not in self.ohlc_columns]
        return self     # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before ConvertUnstableCols is {df.shape}")
        merge_dict = {}
        print_log(f"Number of unstable columns are {len(self.unstable_cols)}")
        for col in self.unstable_cols:
            merge_dict.update({f'CUC_{col}': df[self.basis_column] - df[col]})
            #df[f'PerChg_{col}_{self.periods}_{self.freq}'] =df[col].pct_change(periods=self.periods,fill_method=self.fill_method,limit = self.limit,freq=self.freq)
            print_log(f"CUC_{col} completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after ConvertUnstableCols is {df.shape}") 
        if self.transform_option == 'bint':
            bt_pipe = Pipeline([
                ('bt1', BinningTransform(columns= self.unstable_cols,window=15,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
                ('bt2', BinningTransform(columns= self.unstable_cols,window=30,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
                ('bt3', BinningTransform(columns= self.unstable_cols,window=45,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
                ('bt4', BinningTransform(columns= self.unstable_cols,window=60,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True))
                    ])
        elif self.transform_option == 'rank':
            bt_pipe = Pipeline([
                ('bt1', RollingRank(columns= self.unstable_cols,window=15,min_periods=None,verbose=True)),
                ('bt2', RollingRank(columns= self.unstable_cols,window=30,min_periods=None,verbose=True)),
                ('bt3', RollingRank(columns= self.unstable_cols,window=45,min_periods=None,verbose=True)),
                ('bt4', RollingRank(columns= self.unstable_cols,window=60,min_periods=None,verbose=True))
                    ])
        else:
            bt_pipe = Pipeline([
                ('bt1', PercentageChange_Multiplier(columns= self.unstable_cols,periods=15, fill_method='pad', limit=None, freq=None,verbose=True)),
                ('bt2', PercentageChange_Multiplier(columns= self.unstable_cols,periods=30, fill_method='pad', limit=None, freq=None,verbose=True)),
                ('bt3', PercentageChange_Multiplier(columns= self.unstable_cols,periods=45, fill_method='pad', limit=None, freq=None,verbose=True)),
                ('bt4', PercentageChange_Multiplier(columns= self.unstable_cols,periods=60, fill_method='pad', limit=None, freq=None,verbose=True))
                    ])
        df = bt_pipe.fit_transform(df)
        print_log(f"Shape of dataframe after applying transform in ConvertUnstableCols is {df.shape}") 
        df = df.drop(self.unstable_cols,axis=1)
        print_log(f"Shape of dataframe after dropping unstable columns is {df.shape}")
        return df
    
class feature_mart(DefineConfig):
    def __init__(self, 
                 master_config_path, 
                 master_config_name, 
                 db_conection, 
                 database_path, 
                 train_feature_table=None,
                 train_feature_info_table=None,
                 verbose=True,
                 mode='train'):
        DefineConfig.__init__(self, master_config_path, master_config_name)
        self.db_conection = db_conection
        self.database_path = database_path
        self.mode = mode
        self.ohlc = ddu.load_table_df(
            self.db_conection, table_name=self.ohlc_raw_data_table)
        self.ohlc = self.ohlc.drop_duplicates(subset=['timestamp'], keep='first')

        print_log(f"Shape of OHLC dataframe is {self.ohlc.shape}")
        print(self.ohlc.head())
        self.filter_deta_date = None
        if train_feature_table is not None:
            self.train_feature_table = train_feature_table
            
        if train_feature_info_table is not None:
            self.train_feature_info_table = train_feature_info_table
        print_log(f"feature table is {self.train_feature_table}")
        print_log(f"feature info table is {self.train_feature_info_table}")
        self.first_time_train_feature_info_status = False
        self.delta_len = 0
        self.verbose = verbose
        self.feature_table_setup()
                
    def feature_table_setup(self):
        sql = f"select max(timestamp) as max_timestamp from {self.ohlc_raw_data_table}"
        self.max_date_raw_table = self.db_conection.execute(sql).fetchone()[0]
        sql = f"select min(timestamp) as min_timestamp from {self.ohlc_raw_data_table}"
        self.min_date_raw_table = self.db_conection.execute(sql).fetchone()[0]
        print_log(f"Value of self.max_date_raw_table in {self.ohlc_raw_data_table} is {self.max_date_raw_table}")
        try:
            sql = f"select max(timestamp) as max_timestamp from {self.train_feature_table}"
            self.max_date_feature_table = self.db_conection.execute(sql).fetchone()[0]
            print_log(f"Value of self.max_date_feature_table is {self.max_date_feature_table}")
            sql = f"select min(timestamp) as min_timestamp from {self.train_feature_table}"
            self.min_date_feature_table = self.db_conection.execute(sql).fetchone()[0]
            print_log(f"Value of self.min_date_feature_table is {self.min_date_feature_table}")
        except Exception as e:
            print_log(f"Error encountered while getting max date from {self.train_feature_table} : {e}")
            self.max_date_feature_table = None
            self.min_date_feature_table = None
        if not ddu.check_if_table_exists(self.db_conection, table_name=self.train_feature_table):
            print_log(f"Table {self.train_feature_table} not present")
            ddu.create_table(self.db_conection, table_name=self.train_feature_table, create_table_arg={
                             'replace': True}, df=self.ohlc)
            print_log(f"Table {self.train_feature_table} created")
            self.first_time = True
        else:
            print_log(f"Table {self.train_feature_table} already present")
            self.first_time = False
        if not ddu.check_if_table_exists(self.db_conection, table_name=self.train_feature_info_table):
            create_table_arg = {
                'replace': True, 'table_column_arg': 'function_name VARCHAR ,feature_name VARCHAR, max_date TIMESTAMP,min_date TIMESTAMP,status VARCHAR,column_type VARCHAR,feature_args VARCHAR, updated_timestamp TIMESTAMP'}
            #ddu.create_table(self.db_conection, table_name=self.zip_files_table,
            ddu.create_table(self.db_conection, table_name=self.train_feature_info_table, create_table_arg=create_table_arg, df=None)
            print_log(f"Table {self.train_feature_info_table} created")
            self.first_time_train_feature_info_status = True
        else:
            print_log(f"Table {self.train_feature_info_table} already created")
            self.train_feature_info = ddu.load_table_df(self.db_conection, table_name=self.train_feature_info_table) 
            self.features_from_table = set(self.train_feature_info['feature_name'].tolist())
            print_log(f"Number of already loaded features are {len(self.features_from_table)}")
            self.first_time_train_feature_info_status = False         
        if self.max_date_feature_table is not None and self.first_time is False :
            if self.max_date_feature_table > self.max_date_raw_table:
                delta_df = self.ohlc[self.ohlc['timestamp'] > self.max_date_feature_table]
                self.delta_len = len(delta_df)
                print(f"Shape of Delta dataframe is {delta_df.shape}")
                if self.delta_len > 0:
                    ddu.insert_data(
                        self.db_conection, table_name=self.train_feature_table, insert_arg={}, df=delta_df)
                    
                    #for col in [c for c in delta_df.column.tolist() if c not in ['ticker','timestamp']]:
                    
                    #for col in delta_df.column.tolist():
                    #    sql = f'''UPDATE {self.train_feature_table}
                    #    SET {col} = delta_df.{col}
                    #    FROM delta_df
                    #    WHERE {self.train_feature_table}.{time_stamp_col} = delta_df.{time_stamp_col}
                    #    '''
                    #    print(f"SQL to be run is {sql}")
                    #    self.db_conection.execute(sql)
                else:
                    print_log(f"Delta columns not updated in {self.train_feature_table}")
            self.filter_deta_date = self.max_date_feature_table - timedelta(days=60)
            #self.ohlc = self.ohlc[self.ohlc['timestamp'] > self.filter_deta_date]
            
        #else:
        #    ddu.insert_data(
        #        self.db_conection, table_name=self.train_feature_table, insert_arg={}, df=self.ohlc)

    def convert_df_to_timeseries(self, df):
        df['date_time'] = df['date'].astype(str) + ' ' + df['time']
        df = df.sort_values(by='date_time')
        df.index = df['date_time']
        df = df[['open', 'high', 'low', 'close']]
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        return df

    def label_generator_14class(self, val):
        if val <= 10 and val >= -10:
            return '-10to10'
        elif val > 10 and val <= 20:
            return '10to20'
        elif val > 20 and val <= 40:
            return '20to40'
        elif val > 40 and val <= 60:
            return '40to60'
        elif val > 60 and val <= 80:
            return '60to80'
        elif val > 80 and val <= 100:
            return '80to100'
        elif val > 100:
            return 'above100'
        elif val < -10 and val >= -20:
            return '-10to-20'
        elif val < -20 and val >= -40:
            return '-20to-40'
        elif val < -40 and val >= -60:
            return '-40to-60'
        elif val < -60 and val >= -80:
            return '-60to-80'
        elif val < -80 and val >= -100:
            return '-80to-100'
        elif val < -100:
            return 'below100'
        else:
            return 'unknown'

    def label_generator_9class(self, val):
        if val <= 35 and val >= 0:
            return '-0to35'
        elif val > 35 and val <= 80:
            return '35to80'
        elif val > 80 and val <= 150:
            return '80to150'
        elif val > 150:
            return 'above150'
        elif val > -35 and val <= 0:
            return '0to-35'
        elif val > -80 and val <= -35:
            return '-35to-80'
        elif val > -150 and val <= -80:
            return '-80to-150'
        elif val < -150:
            return 'below150'
        else:
            return 'unknown'

    def label_generator_7class(self, val):
        if val <= 30 and val >= 0:
            return '-0to30'
        elif val > 30 and val <= 80:
            return '30to80'
        elif val > 80:
            return 'above80'
        elif val > -30 and val <= 0:
            return '0to-30'
        elif val > -80 and val <= -30:
            return '-30to-80'
        elif val <= -80:
            return 'below80'
        else:
            return 'unknown'

    def label_generator_4class(self, val):
        if val <= 30 and val >= -30:
            return 'neutral'
        elif val > 30:
            return 'call'
        elif val < -30:
            return 'put'
        else:
            return 'unknown'


    def label_generator_4class_50points(self, val):
        if val <= 50 and val >= -50:
            return 'neutral'
        elif val > 50:
            return 'call'
        elif val < -50:
            return 'put'
        else:
            return 'unknown'
 
    def label_generator_3class_50points_no_neutral(self, val):
        if val > 50:
            return 'call'
        elif val < -50:
            return 'put'
        else:
            return 'unknown'

    def label_generator_3class_20points_no_neutral(self, val):
        if val > 20:
            return 'call'
        elif val < -20:
            return 'put'
        else:
            return 'unknown'
                  
    def label_generator_4class_mod1(self, val):
        if val <= 20 and val >= -20:
            return 'neutral'
        elif val > 20:
            return 'call'
        elif val < -20:
            return 'put'
        else:
            return 'unknown'

    def label_generator_4class_mod2(self, val):
        if val <= 20 and val >= -20:
            return 'neutral'
        elif val > 30:
            return 'call'
        elif val < -30:
            return 'put'
        else:
            return 'unknown'
               
    def get_duck_db_dtype(self, obj):
        if is_numeric_dtype(obj):
            return 'DOUBLE'
        if is_string_dtype(obj):
            return 'VARCHAR'
        if is_bool_dtype(obj):
            return 'BOOLEAN'
        if is_datetime64_any_dtype(obj):
            return 'TIMESTAMP'
        else:
            return 'DOUBLE'
    
    def save_feature_info(self,function_name,feature_name,max_date,min_date,status,feature_args,column_type='feature'):
        feature_args = {x:y if y is not None else str(y) for x,y in feature_args.items() }
        if feature_name in ['close','open','ticker','timestamp','high','low']:
            column_type = 'ohlc_column'
        if self.first_time_train_feature_info_status is False and feature_name in self.features_from_table:
            sql = f'''
                UPDATE {self.train_feature_info_table}
                SET 
                function_name = '{function_name}',
                feature_name = '{feature_name}',
                max_date = '{str(max_date)}',
                min_date = '{str(min_date)}',
                status = '{status}',
                column_type = '{column_type}',
                feature_args = {feature_args},
                updated_timestamp = '{dt.datetime.now()}'              
                WHERE feature_name = '{feature_name}'
                '''
            print_log(f"SQL to be run is {sql}")
            self.db_conection.execute(sql)

        else:
            curr_dt = str(dt.datetime.now())
            sql_dict = {"function_name":function_name,"feature_name":feature_name,"max_date":str(max_date),
                        "min_date":str(min_date),"status":status,"column_type":column_type,"feature_args":feature_args,
                        "updated_timestamp":curr_dt}
            sql = f'''
            insert into {self.train_feature_info_table}
            select 
            tab.function_name,
            tab.feature_name,
            tab.max_date,
            tab.min_date,
            tab.status,
            tab.column_type,
            tab.feature_args,
            tab.updated_timestamp from
            (select {sql_dict} as tab)
            '''
            print_log(f"SQL to be run is {sql}")
            self.db_conection.execute(sql)
            #insert_arg = {"insert_values":f'"{function_name}","{feature_name}","{status}","{feature_args}","{dt.datetime.now()}"'}
            #ddu.insert_data(self.db_conection,self.train_feature_info_table,insert_arg)
    
    def save_feature_info_v2(self,function_name,feature_name,max_date,min_date,status,feature_args,column_type='feature'):
        if self.mode == 'train':
            curr_dt = str(dt.datetime.now())
            sql_dict = {"function_name":function_name,"feature_name":feature_name,"max_date":str(max_date),
                "min_date":str(min_date),"status":status,"column_type":column_type,"feature_args":feature_args,
                "updated_timestamp":curr_dt}
            update_where_expr = f"feature_name = '{feature_name}'"
            self.update_insert(sql_dict = sql_dict,
                            table_name = self.train_feature_info_table,
                            update_where_expr=update_where_expr)
        else:
            print_log(f"Not saving any data in {self.train_feature_info_table} table since mode is {self.mode}")
        
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
        print_log(f"Update sql is {sql}")
        t = self.db_conection.execute(sql)
        update_cnt = t.fetchall()
        update_cnt = update_cnt[0][0] if len(update_cnt) > 0 else 0
        print_log(f"Update count is {update_cnt}")
        if update_cnt == 0:
            sql_dict_updated = {i: ( str(j) if j is None else j) for i,j in sql_dict.items() }
            insert_col_expr = [f"tab.{i}" for i,j in sql_dict_updated.items()]
            insert_col_expr = ",".join(insert_col_expr)
            sql = f'''
            insert into {table_name}
            select {insert_col_expr} from
            (select {sql_dict_updated} as tab)
            '''
            print_log(f"Insert sql is {sql}")
            a = self.db_conection.execute(sql)
            insert_cnt = a.fetchall()
            insert_cnt = insert_cnt[0][0] if len(insert_cnt) > 0 else 0
            print_log(f"Insert count is {insert_cnt}")

    def check_and_create_table(self,table_name,create_table_arg):
        if not ddu.check_if_table_exists(self.db_conection, table_name=table_name):
            ddu.create_table(self.db_conection, table_name=table_name, create_table_arg=create_table_arg, df=None)
    
    def check_and_add_columns(self,df,table_name):
        for column_name in df.columns.tolist():
            data_type = self.get_duck_db_dtype(df[column_name]) 
            if not ddu.check_if_table_and_column_exists(self.db_conection, self.train_feature_table, column_name):
                print_log(f"Column {column_name} not present in dataframe")
                alter_arg = {'alter_type': 'add_column',
                            'column_name': column_name, 'data_type': data_type}
                ddu.alter_table(self.db_conection,
                                table_name, alter_arg=alter_arg)
                                  
    def create_column_and_save_to_table(self, time_stamp_col, data,exclude_cols=None):
        print_log(f"Shape of the dataframe before join {data.shape}")
        column_status= False
        if len(data) > 0:
            data.columns = data.columns.str.replace('[#,@,&,-,.,"]', '')
            data = data.rename(columns=lambda x: x.strip())
            #data.to_csv('../../notebook/new/test.csv')
            data = data.reset_index(drop=True)
            #print_log(f"Shape of the dataframe before join {data.shape}")
            column_names = [col for col in data.columns.tolist() if col != time_stamp_col]
            if exclude_cols is not None:
                column_names = [col for col in data.columns.tolist() if col not in exclude_cols]

            for column_name in column_names:
                print_log(f"Data type of {column_name} : {data[column_name].dtype}")
                data_type = self.get_duck_db_dtype(data[column_name])
                print_log(f"Duckdb datatype of {column_name} : {data_type}")
                if not ddu.check_if_table_and_column_exists(self.db_conection, self.train_feature_table, column_name):
                    alter_arg = {'alter_type': 'add_column',
                                'column_name': column_name, 'data_type': data_type}
                    ddu.alter_table(self.db_conection,
                                    self.train_feature_table, alter_arg=alter_arg)
                    column_status = True

                if self.max_date_feature_table  is None or column_status is True:
                    sql = f'''UPDATE {self.train_feature_table}
                    SET {column_name} = data.{column_name}
                    FROM data
                    WHERE {self.train_feature_table}.{time_stamp_col} = data.{time_stamp_col}
                    '''
                    print_log(f"SQL to be run is {sql}")
                    self.db_conection.execute(sql)
                else:
                    sql = f'''UPDATE {self.train_feature_table}
                    SET {column_name} = data.{column_name}
                    FROM data
                    WHERE {self.train_feature_table}.{time_stamp_col} = data.{time_stamp_col} and 
                    {self.train_feature_table}.{time_stamp_col} > '{self.max_date_feature_table}'
                    '''
                    print_log(f"SQL to be run is {sql}")
                    self.db_conection.execute(sql)  
        else:
            print_log(f"Not saving any columns since shape of data is {data.shape}")            

    def create_column_and_save_to_table_v2(self, time_stamp_col, data):
        print_log(f"Shape of the dataframe before join {data.shape}")
        if len(data) > 0 and self.mode == 'train':
            data.columns = data.columns.str.replace('[#,@,&,-,.,"]', '')
            data = data.rename(columns=lambda x: x.strip())
            data = data.reset_index(drop=True)
            
            column_names = [col for col in data.columns.tolist() if col != time_stamp_col]
            column_names = [col for col in data.columns.tolist() if col not in ['close','open','ticker','timestamp','high','low']]
            try:
                features_with_date = self.get_features_with_load_date(feature_type=None,feature_cols=column_names,
                                                                    return_load_date_for_ohlc=False)
            except Exception as e:
                print_log(f"Unable to fetch the dates : {e}")
                features_with_date = {col:None for col in column_names}
                
            for feature_name, load_date in features_with_date.items():
                print_log(f"Data type of {feature_name} : {data[feature_name].dtype}")
                data_type = self.get_duck_db_dtype(data[feature_name])
                print_log(f"Duckdb datatype of {feature_name} : {data_type}")
                
                if load_date is None:
                    alter_arg = {'alter_type': 'add_column',
                                'column_name': feature_name, 'data_type': data_type}
                    ddu.alter_table(self.db_conection,
                                    self.train_feature_table, alter_arg=alter_arg)
                    
                    sql = f'''UPDATE {self.train_feature_table}
                    SET {feature_name} = data.{feature_name}
                    FROM data
                    WHERE {self.train_feature_table}.{time_stamp_col} = data.{time_stamp_col} 
                    '''
                    print_log(f"SQL to be run is {sql}")
                    self.db_conection.execute(sql) 
                else:
                    sql = f'''UPDATE {self.train_feature_table}
                    SET {feature_name} = data.{feature_name}
                    FROM data
                    WHERE {self.train_feature_table}.{time_stamp_col} = data.{time_stamp_col} and 
                    {self.train_feature_table}.{time_stamp_col} > '{str(load_date)}'
                    '''
                    print_log(f"SQL to be run is {sql}")
                    self.db_conection.execute(sql)  
        else:
            print_log(f"Not saving any columns since shape of data is {data.shape} and mode is {mode}")  
            
    def get_ohlc_df(self, df=None,column_name = None,force_delta=False):
        if df is None:
            #df = self.ohlc.copy()
            df = self.get_ohlc_based_on_existance(column_name,force_delta)
            
        if df is not None:
            df = self.create_timestamp_index_df(df)
            print_log(f"Shape of OHLC dataframe is {df.shape}")
        return df

    def create_timestamp_index_df(self,df=None):
        if df is None:
            df = self.ohlc.copy()
        df.index = df['timestamp']
        df.index = pd.DatetimeIndex(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        return df
            
    def get_features_with_load_date(self,feature_type='unstable_column',feature_cols=None,return_load_date_for_ohlc=True):
        if feature_type is None: 
            sql = f"select * from {self.train_feature_info_table}"
        else:
            sql = f"select * from {self.train_feature_info_table} where column_type = '{feature_type}'"
        info = ddu.load_table_df(self.db_conection,table_name=None,column_names=None,filter=None,load_sql=sql)
        if feature_cols is None:
            feature_cols = info['feature_name'].tolist()
            feature_cols = [col.strip() for col in feature_cols]
        ret_dict = {}
        for col in feature_cols:
            ret_dict.update({col:self.get_feature_date(col,info,return_load_date_for_ohlc)})
        return ret_dict
    
    def get_feature_date(self,col_name,info_df=None,return_load_date_for_ohlc=True):
        try:
            if info_df is None:
                sql = f"select * from {self.train_feature_info_table}"
                info_df = ddu.load_table_df(self.db_conection,table_name=None,column_names=None,filter=None,load_sql=sql)
            f_list = info_df['feature_name'].tolist()
            f_list = [col.strip() for col in f_list]
            d_list = info_df['max_date'].tolist()
            ret_dict = {i:j for i,j in zip(f_list,d_list)}
            max_col_date = ret_dict.get(col_name)
            if max_col_date is not None:
                if max_col_date < self.max_date_raw_table:
                    if return_load_date_for_ohlc:
                        return self.max_date_raw_table - dt.timedelta(days=60)
                    else:
                        return max_col_date
                else:
                    return self.max_date_raw_table
            else:
                return None
        except Exception as e:
            print_log(f"Error found during feature lookup : {e}")
            return None
        
    def get_ohlc_based_on_existance(self, column_name=None,force_delta=False):
        column_status = False
        if column_name is not None:
            column_status = ddu.check_if_table_and_column_exists(self.db_conection, self.train_feature_table, column_name)
            print_log(f"Column Status for {column_name} is {column_status}")
        if not column_status and force_delta is False:
            return self.ohlc.copy()
        print_log(f"Length of delta df is {self.delta_len}")
        if self.delta_len > 0:
            print_log(f"Delta df length is {self.delta_len}")
            return self.ohlc[self.ohlc['timestamp'] > self.filter_deta_date]
        else:
            return None
    
    def get_ohlc_df_v2(self, df=None,column_name = None):
        df = self.get_ohlc_based_on_existance_v2(df,column_name)
        print_log(f"Shape of extracted OHLC dataframe is {df.shape}")
        if df is not None:
            df = self.create_timestamp_index_df(df)
        if len(df)>0:
            return df
        else:   
            return None
    
    def get_ohlc_based_on_existance_v2(self, df=None,column_name=None):
        if df is not None and column_name is not None:
            column_load_date = self.get_feature_date(col_name=column_name,info_df=None,return_load_date_for_ohlc=True)
            print_log(f"column_load_date is {column_load_date}")
            if column_load_date is not None:
                return df[df['timestamp'] > column_load_date]
            else:
                return df
        elif df is None and column_name is not None:
            column_load_date = self.get_feature_date(col_name=column_name,info_df=None,return_load_date_for_ohlc=True)
            if column_load_date is not None:
                return self.ohlc[self.ohlc['timestamp'] > column_load_date]
            else:
                return self.ohlc.copy()
        else:
            return  self.ohlc.copy()  
        
    def remove_ohlc_cols(self, df,col_names=None):
        if col_names is not None:
            return df[col_names]
        else:
            ohlc_cols = ['open', 'close', 'high', 'low', 'ticker']
            all_cols = [col for col in df.columns.tolist() if col not in ohlc_cols]
            return df[all_cols]

    def keep_timestamp_feature_cols(self, df, cols):
        final_cols = ['timestamp'] + cols
        return df[final_cols]

    def get_min_max_date(self,tmpdf):
        return max(tmpdf['timestamp']),min(tmpdf['timestamp'])
    
    def label_creator(self, func_dict_args, tmpdf=None, return_df=False):
        # label_creator_args = {'freq':'1min','shift':-15,'shift_column':'close','generator_function_name':'label_generator_4class'}
        print_log("*"*100)
        print_log(f"label_creator called with arguments {func_dict_args}")
        label_name = f"label_{func_dict_args['shift']}_{func_dict_args['freq']}_{func_dict_args['shift_column']}__{func_dict_args['generator_function_name']}".replace(
            "-", '_minus_')
        if tmpdf is None:
            tmpdf = self.get_ohlc_df(column_name=label_name)
            if tmpdf is None:
                return
        apply_func = getattr(self, func_dict_args['generator_function_name'])
        tmpdf[label_name] = tmpdf.shift(func_dict_args['shift'], freq=func_dict_args['freq'])[func_dict_args['shift_column']].subtract(
            tmpdf[func_dict_args['shift_column']]).apply(apply_func)
        tmpdf = self.remove_ohlc_cols(tmpdf)
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        self.save_feature_info(function_name='label_creator',max_date=max_date,min_date=min_date,feature_name=label_name,status='saved',feature_args=func_dict_args,column_type='label')
        if return_df:
            return tmpdf
        del tmpdf

    def create_technical_indicator_using_pandasta(self, func_dict_args, tmpdf=None, return_df=False):
        # create_technical_indicator_using_pandasta_args= {'exclude':["jma","pvo","vwap","vwma","ad","adosc","aobv","cmf","efi","eom","kvo","mfi","nvi","obv","pvi","pvol","pvr","pvt"]}
        import pandas_ta as ta
        print_log("*"*100)
        print_log(
            f"create_technical_indicator_using_pandasta called with arguments {func_dict_args}")
        if tmpdf is None:
            tmpdf = self.get_ohlc_df()
            if tmpdf is None:
                return
        tmpdf.ta.strategy(
            exclude=func_dict_args['exclude'], verbose=self.verbose, timed=True,lookahead=False)
        tmpdf = self.remove_ohlc_cols(tmpdf)
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        for col in tmpdf.columns.tolist():
            self.save_feature_info(function_name='create_technical_indicator_using_pandasta',feature_name=col,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
        if return_df:
            return tmpdf
        del tmpdf

    def create_technical_indicator_using_pandasta_list(self, func_dict_args, tmpdf=None, return_df=False):
        # create_technical_indicator_using_pandasta_args= {'exclude':["jma","pvo","vwap","vwma","ad","adosc","aobv","cmf","efi","eom","kvo","mfi","nvi","obv","pvi","pvol","pvr","pvt"]}
        import pandas_ta as ta
        print_log("*"*100)
        print_log(f"create_technical_indicator_using_pandasta_list called with arguments {func_dict_args}")
        #self.pandasta_pipe = func_dict_args.get('pandasta_pipe')
        self.technical_indicator_pipeline = self.technical_indicator_pipeline if func_dict_args['technical_indicator_pipeline'] is None else func_dict_args['technical_indicator_pipeline']
        df_list = []
        if tmpdf is None:
            tmpdf = self.get_ohlc_df()
            if tmpdf is None:
                return
        pipe_config = {}
        for pipe in self.technical_indicator_pipeline:
            print_log(f"Running technical indicator pipeline {pipe}")
            pipe_config.update({'name':f'pipe_desc_{pipe}'})
            pipe_config.update({'ta': getattr(self,pipe)})
            print_log(f"Pipeline configuration is {pipe_config}")
            pipe_desc = ta.Strategy(**pipe_config)
            tmpdf.ta.strategy(pipe_desc,
                            exclude=func_dict_args['exclude'],
                            verbose=self.verbose, 
                            timed=True,
                            lookahead=False)
            func_dict_args['pipe_config'] = pipe_config
            func_dict_args['strategy_config'] = {'exclude':func_dict_args['exclude'],'verbose':self.verbose,'timed':True,'lookahead':False}
            tmpdf = self.remove_ohlc_cols(tmpdf)
            max_date,min_date = self.get_min_max_date(tmpdf)
            self.create_column_and_save_to_table(time_stamp_col='timestamp', data=tmpdf)
            for col in tmpdf.columns.tolist():
                self.save_feature_info(function_name='create_technical_indicator_using_pandasta_list',feature_name=col,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            if return_df:
                df_list.append(tmpdf)
        if return_df:
            return df_list
        del tmpdf

    def create_technical_indicator_using_pandasta_list_one(self, func_dict_args, tmpdf=None, return_df=False):
        # create_technical_indicator_using_pandasta_args= {'exclude':["jma","pvo","vwap","vwma","ad","adosc","aobv","cmf","efi","eom","kvo","mfi","nvi","obv","pvi","pvol","pvr","pvt"]}
        import pandas_ta as ta
        print_log("*"*100)
        print_log(f"create_technical_indicator_using_pandasta_list_one called with arguments {func_dict_args}")
        #self.pandasta_pipe = func_dict_args.get('pandasta_pipe')
        self.technical_indicator_pipeline = self.technical_indicator_pipeline if func_dict_args['technical_indicator_pipeline'] is None else func_dict_args['technical_indicator_pipeline']
        df_list = []
        if tmpdf is None:
            tmpdf = self.get_ohlc_df()
            if tmpdf is None:
                return
        
        for pipe in self.technical_indicator_pipeline:
            print_log(f"Running technical indicator pipeline {pipe}")
            
            i = 1
            pipe_config = {}
            for pipe_delta in getattr(self,pipe):
                if tmpdf is None:
                    tmpdf_copy = self.get_ohlc_df()
                    if tmpdf is None:
                        return
                else:
                    tmpdf_copy = tmpdf.copy()
                pipe_config.update({'name':f'pipe_desc_{pipe}_{i}'})
                pipe_config.update({'ta': [pipe_delta]})
                i = i+1
                print_log(f"Pipeline configuration is {pipe_config}")
                pipe_desc = ta.Strategy(**pipe_config)
                tmpdf_copy.ta.strategy(pipe_desc,
                                exclude=func_dict_args['exclude'],
                                verbose=self.verbose, 
                                timed=True,
                                lookahead=False)
                func_dict_args['pipe_config'] = pipe_config
                func_dict_args['strategy_config'] = {'exclude':func_dict_args['exclude'],
                                                     'verbose':self.verbose,'timed':True,
                                                     'lookahead':False}
                tmpdf_copy = self.remove_ohlc_cols(tmpdf_copy)
                max_date,min_date = self.get_min_max_date(tmpdf_copy)
                self.create_column_and_save_to_table(time_stamp_col='timestamp', data=tmpdf_copy)
                for col in tmpdf_copy.columns.tolist():
                    self.save_feature_info(function_name='create_technical_indicator_using_pandasta_list_one',
                                           feature_name=col,
                                           max_date=max_date,
                                           min_date=min_date,
                                           status='saved',
                                           feature_args=func_dict_args)
                if return_df:
                    df_list.append(tmpdf_copy)
                del tmpdf_copy
        if return_df:
            return df_list
        #del tmpdf_copy
        
    def create_technical_indicator_using_signals(self, func_dict_args, tmpdf=None, return_df=False):
        # create_technical_indicator_using_signals_args= {'method_type':['volumn_','volatile_','transform_','cycle_','pattern_','stats_','math_','overlap_']}
        import pandas_ta as ta
        print_log("*"*100)
        print_log(
            f"create_technical_indicator_using_signals called with arguments {func_dict_args}")
        if tmpdf is None:
            tmpdf = self.get_ohlc_df()
            if tmpdf is None:
                return
        all_methods = []
        a = dict(Signals.__dict__)
        for a1, a2 in a.items():
            all_methods.append(a1)
        all_methods = [m1 for m1, m2 in a.items() if m1[:1] != '_']
        all_methods = [m for m in all_methods for mt in func_dict_args.get(
            'method_type') if mt in m]

        sig = Signals(tmpdf)
        methods_run = []
        methods_notrun = []
        for f in all_methods:
            try:
                exec(f'sig.{f}()')
                methods_run.append(f)
            except Exception as e1:
                print_log(f"Function {f} was unable to run, Error is {e1}")
                methods_notrun.append(f)

        tmpdf = self.remove_ohlc_cols(tmpdf)
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        for col in tmpdf.columns.tolist():
            self.save_feature_info(function_name='create_technical_indicator_using_signals',feature_name=col,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
 
        if return_df:
            return tmpdf
        del tmpdf

    def create_technical_indicator_using_ta(self, func_dict_args, tmpdf=None, return_df=False):
        # open='open',high='high',low='low',close='close',volume='volume',vectorized=True,fillna=False,colprefix='ta',volume_ta=True,volatility_ta=True,trend_ta=True,momentum_ta=True,others_ta=True,verbose=True
        print_log("*"*100)
        print_log(
            f"create_technical_indicator_using_ta called with arguments {func_dict_args}")
        if tmpdf is None:
            tmpdf = self.get_ohlc_df()
            if tmpdf is None:
                return
        tmpdf = add_all_ta_features(
            tmpdf,
            open=func_dict_args.get('open'),
            high=func_dict_args.get('high'),
            low=func_dict_args.get('low'),
            close=func_dict_args.get('close'),
            volume=func_dict_args.get('volume'),
            fillna=func_dict_args.get('fillna'),
            colprefix=func_dict_args.get('colprefix'),
            vectorized=func_dict_args.get('vectorized'),
            volume_ta=func_dict_args.get('volume_ta'),
            volatility_ta=func_dict_args.get('volatility_ta'),
            trend_ta=func_dict_args.get('trend_ta'),
            momentum_ta=func_dict_args.get('momentum_ta'),
            others_ta=func_dict_args.get('others_ta'),
        )
        tmpdf = self.remove_ohlc_cols(tmpdf)
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        for col in tmpdf.columns.tolist():
            self.save_feature_info(function_name='create_technical_indicator_using_ta',feature_name=col,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)

        if return_df:
            return tmpdf
        del tmpdf
            

    def normalize_dataset(self, df, func_dict_args):
        #  column_pattern = [],columns = [],impute_values=False,impute_type = 'categorical',convert_to_floats = False,arbitrary_impute_variable=99,drop_na_col=False,drop_na_rows=False,
        #  fillna = False,fillna_method = 'bfill',fill_index=False
        print_log("*"*100)
        print_log(f"normalize_dataset called with arguments {func_dict_args}")
        if len(func_dict_args.get('columns')) == 0:
            func_dict_args['columns'] = [m for m in df.columns.tolist(
            ) for mt in func_dict_args.get('column_pattern') if mt in m]
            func_dict_args['columns'] = list(
                set(func_dict_args.get('columns')))

        info_list = []
        df = convert_todate_deduplicate(df)
        if func_dict_args.get('convert_to_floats'):
            for col in func_dict_args['columns']:
                df[col] = df[col].astype('float')
                info_list.append('convert_to_floats')
        if func_dict_args.get('fill_index'):
            df = df.reindex(pd.date_range(
                min(df.index), max(df.index), freq='1min'))
            df = df.resample('1min').ffill()
        if func_dict_args.get('impute_values'):
            from sklearn.pipeline import Pipeline
            if func_dict_args.get('impute_type') == 'mean_median_imputer':
                imputer = MeanMedianImputer(
                    imputation_method='median', variables=func_dict_args['columns'])
                info_list.append('mean_median_imputer')
            elif func_dict_args.get('impute_type') == 'categorical':
                imputer = CategoricalImputer(
                    variables=func_dict_args['columns'])
                info_list.append('categorical')
            elif func_dict_args.get('impute_type') == 'arbitrary':
                if isinstance(func_dict_args.get('arbitrary_impute_variable'), dict):
                    imputer = ArbitraryNumberImputer(
                        imputer_dict=func_dict_args.get('arbitrary_impute_variable'))
                else:
                    imputer = ArbitraryNumberImputer(
                        variables=func_dict_args['columns'], arbitrary_number=func_dict_args.get('arbitrary_number'))
                info_list.append('arbitrary')
            else:
                imputer = CategoricalImputer(
                    variables=func_dict_args['columns'])
                info_list.append('categorical')
            imputer.fit(df)
            df = imputer.transform(df)
        if func_dict_args.get('fillna'):
            df = df.fillna(method=func_dict_args.get('fillna_method'))
            info_list.append('fillna')
        if func_dict_args.get('drop_na_col'):
            imputer = DropMissingData(missing_only=True)
            imputer.fit(df)
            df = imputer.transform(df)
            info_list.append('drop_na_col')
        if func_dict_args.get('drop_na_rows'):
            df = df.dropna(axis=0)
            info_list.append('drop_na_rows')
        df = df.sort_index()
        return df

    def rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def last_tick_greater_values_count(self, func_dict_args, tmpdf=None, return_df=False):
        # last_tick_greater_values_count_args = {'column':col,'last_ticks': tick}
        print_log("*"*100)
        print_log(
            f"last_tick_greater_values_count called with arguments {func_dict_args}")
        feature_name = f"ROLLTGC_{func_dict_args.get('column')}_{func_dict_args.get('last_ticks')}".replace(
            "-", '_minus_')
        #col_list = ['timestamp',feature_name]
        if tmpdf is None:
            tmpdf = self.get_ohlc_df(column_name=feature_name)
            if tmpdf is None:
                return
        x = np.concatenate([[np.nan] * (func_dict_args.get('last_ticks')),
                           tmpdf[func_dict_args.get('column')].values])
        arr = self.rolling_window(x, func_dict_args.get('last_ticks') + 1)
        tmpdf[feature_name] = (arr[:, :-1] > arr[:, [-1]]).sum(axis=1)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=[feature_name])
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        self.save_feature_info(function_name='last_tick_greater_values_count',
                               feature_name=feature_name,max_date=max_date,min_date=min_date,status='saved',
                               feature_args=func_dict_args)
        if return_df:
            return tmpdf
        del tmpdf

    def attach_ohlc(self, func_dict_args, tmpdf=None, return_df=False):
        # last_tick_greater_values_count_args = {'column':col,'last_ticks': tick}
        print_log("*"*100)
        print_log(
            f"last_tick_greater_values_count called with arguments {func_dict_args}")
        #feature_name = f"ROLLTGC_{func_dict_args.get('column')}_{func_dict_args.get('last_ticks')}".replace(
        #    "-", '_minus_')
        if tmpdf is None:
            tmpdf = self.get_ohlc_df()
            if tmpdf is None:
                return
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        max_date,min_date = self.get_min_max_date(tmpdf)
        for col in self.ohlc_column:
            self.save_feature_info(function_name='attach_ohlc',feature_name=col,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args,column_type='ohlc_column')
        if return_df:
            return tmpdf
        del tmpdf
        
    def price_last_tick_breach_count(self, func_dict_args, tmpdf=None, return_df=False):
        # price_last_tick_breach_count_args = {'column_name':col,'breach_type': 'morethan','last_ticks':tick}
        print_log("*"*100)
        column = func_dict_args.get('column_name')
        breach_type = func_dict_args.get('breach_type')
        last_ticks = func_dict_args.get('last_ticks')
        
        print_log(f"price_last_tick_breach_count called with arguments {func_dict_args}")
        feature_name = f"ROLLTBC_{breach_type}_{column}_{last_ticks}".replace("-", '_minus_')
        if tmpdf is None:
            tmpdf = self.get_ohlc_df(column_name=feature_name)
            if tmpdf is None:
                return
        
        if breach_type == 'morethan':
            tmpdf[feature_name] = tmpdf[column].rolling(last_ticks, min_periods=1).apply(lambda x: (x[-1] > x[:-1]).sum()).fillna(0)
        elif breach_type== 'lessthan':
            tmpdf[feature_name] = tmpdf[column].rolling(last_ticks, min_periods=1).apply(lambda x: (x[-1] < x[:-1]).sum()).fillna(0)
        elif breach_type == 'mean':
            tmpdf[feature_name] = tmpdf[column].rolling(last_ticks, min_periods=1).apply(lambda x: (x > x[:].mean()).sum()).fillna(0).astype(int)
        elif breach_type == 'min':
            tmpdf[feature_name] = tmpdf[column].rolling(last_ticks, min_periods=1).apply(lambda x: (x > x[:].min()).sum()).fillna(0).astype(int)
        elif breach_type == 'max':
            tmpdf[feature_name] = tmpdf[column].rolling(last_ticks, min_periods=1).apply(lambda x: (x > x[:].max()).sum()).fillna(0).astype(int)
        elif breach_type == 'median':
            tmpdf[feature_name] = tmpdf[column].rolling(last_ticks, min_periods=1).apply(lambda x: (x > x[:].median()).sum()).fillna(0).astype(int)
        elif breach_type == '10thquantile':
            tmpdf[feature_name] = tmpdf[column].rolling(last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.1)).sum()).fillna(0).astype(int)
        elif breach_type == '25thquantile':
            tmpdf[feature_name] = tmpdf[column].rolling(last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.25)).sum()).fillna(0).astype(int)
        elif breach_type == '75thquantile':
            tmpdf[feature_name] = tmpdf[column].rolling(last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.75)).sum()).fillna(0).astype(int)
        elif breach_type  == '95thquantile':
            tmpdf[feature_name] = tmpdf[column].rolling(last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.95)).sum()).fillna(0).astype(int)
        else:
            tmpdf[feature_name] = (tmpdf[column].rolling(last_ticks, min_periods=1)
                                   .apply(lambda x: (x[-1] > x[:-1]).mean())
                                   .astype(int))
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=[feature_name])
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.save_feature_info(function_name='price_last_tick_breach_count',feature_name=feature_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)

        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        if return_df:
            return tmpdf
        del tmpdf

    def rolling_values(self, func_dict_args, tmpdf=None, return_df=False):
        # rolling_values_args = {'column_name':col,'aggs':['mean','max'],'last_ticks':tick,'oper' : ['-','=']}
        print_log("*"*100)
        print_log(f"rolling_values called with arguments {func_dict_args}")
        column = func_dict_args.get('column_name')
        last_ticks = func_dict_args.get('last_ticks')
        aggs = func_dict_args.get('aggs')
        oper = func_dict_args.get('oper')
        feature_name = f"ROLLVAL{column}_{'_'.join(last_ticks)}_{'_'.join(aggs)}".replace(
            "-", '_minus_')
        if tmpdf is None:
            tmpdf = self.get_ohlc_df(column_name=feature_name)
            if tmpdf is None:
                return

        eval_stmt = ''
        for lt, oper, agg in zip(last_ticks, oper, aggs):
            tmpst = f"tmpdf['{column}'].rolling('{lt}', min_periods=1).{agg}() {oper}"
            eval_stmt = eval_stmt + tmpst

        print(eval_stmt[:-1])
        tmpdf[feature_name] = eval(eval_stmt[:-1])
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=[feature_name])
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        self.save_feature_info(function_name='rolling_values',feature_name=feature_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
        if return_df:
            return tmpdf
        del tmpdf

    def price_data_range_hour(self, func_dict_args, tmpdf=None, return_df=False):
        # price_data_range_hour_args = {'first_col': 'high', 'second_col': 'low', 'hour_range': hr, 'range_type': rng}
        print_log("*"*100)
        print_log(
            f"price_data_range_hour called with arguments {func_dict_args}")
        r1 = func_dict_args.get('hour_range')[0]
        r2 = func_dict_args.get('hour_range')[1]
        first_col = func_dict_args.get('first_col')
        second_col = func_dict_args.get('second_col')
        range_type = func_dict_args.get('range_type')
        feature_name = f"ROLLPDR_{first_col}_{second_col}_{range_type}_{r1.replace(':','')}_{r2.replace(':','')}".replace(
            "-", '_minus_')
        if tmpdf is None:
            tmpdf = self.get_ohlc_df(column_name=feature_name)
            if tmpdf is None:
                return
        if range_type == 'price_range':
            tmpdf[feature_name] = tmpdf[first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max(
            ) - tmpdf[second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
        elif range_type == 'price_deviation_max_first_col':
            tmpdf[feature_name] = tmpdf[first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean(
            ) - tmpdf[first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max()
        elif range_type == 'price_deviation_min_first_col':
            tmpdf[feature_name] = tmpdf[first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean(
            ) - tmpdf[first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
        elif range_type == 'price_deviation_max_second_col':
            tmpdf[feature_name] = tmpdf[second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean(
            ) - tmpdf[second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max()
        elif range_type == 'price_deviation_min_second_col':
            tmpdf[feature_name] = tmpdf[second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean(
            ) - tmpdf[second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
        else:
            tmpdf[feature_name] = tmpdf[first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max(
            ) - tmpdf[second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
        tmpdf[feature_name] = tmpdf[feature_name].fillna(method='ffill')
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=[feature_name])
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        self.save_feature_info(function_name='price_data_range_hour',feature_name=feature_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
        if return_df:
            return tmpdf
        del tmpdf

    def price_velocity_v2(self, func_dict_args, tmpdf=None, return_df=False):
        # price_velocity_args = {'freq': frq, 'shift': sft, 'shift_column': 'close'}
        print_log("*"*100)
        print_log(f"price_velocity called with arguments {func_dict_args}")
        freq = func_dict_args.get('freq')
        shift = func_dict_args.get('shift')
        shift_column = func_dict_args.get('shift_column')
        if freq is not None:
            col_name = f'ROLLPVR2_{shift_column}_{freq}_{shift}'.replace(
                "-", '_minus_')
            if tmpdf is None:
                tmpdf = self.get_ohlc_df(column_name=col_name)
                if tmpdf is None:
                    return
            tmpdf[col_name] = tmpdf.shift(
                shift, freq=func_dict_args.get('freq'))[shift_column]
        else:
            col_name = f'ROLLPVR2_{shift_column}_{shift}'
            if tmpdf is None:
                tmpdf = self.get_ohlc_df(column_name=col_name)
                if tmpdf is None:
                    return
            tmpdf[col_name] = tmpdf.shift(shift)[shift_column]
        tmpdf[col_name] = tmpdf[shift_column] - tmpdf[col_name]
        tmpdf[col_name] = tmpdf[col_name].round(3)
        print_log(f"price_velocity : {col_name} created")
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=[col_name])
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        self.save_feature_info(function_name='price_velocity_v2',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
        if return_df:
            return tmpdf
        del tmpdf

    def price_velocity(self, func_dict_args, tmpdf=None, return_df=False):
        # price_velocity_args = {'freq': frq, 'shift': sft, 'shift_column': 'close'}
        print_log("*"*100)
        print_log(f"price_velocity called with arguments {func_dict_args}")
        freq = func_dict_args.get('freq')
        shift = func_dict_args.get('shift')
        shift_column = func_dict_args.get('shift_column')
        if freq is not None:
            col_name = f'ROLLPVC_{shift_column}_{freq}_{shift}'.replace(
                "-", '_minus_')
            if tmpdf is None:
                tmpdf = self.get_ohlc_df(column_name=col_name)
                if tmpdf is None:
                    return
            tmpdf[col_name] = tmpdf[shift_column].subtract(tmpdf.shift(shift, freq=freq)[shift_column])
        else:
            col_name = f'ROLLPVC_{shift_column}_{shift}'.replace(
                "-", '_minus_')
            if tmpdf is None:
                tmpdf = self.get_ohlc_df(column_name=col_name)
                if tmpdf is None:
                    return
            tmpdf[col_name] = tmpdf[shift_column].subtract(tmpdf.shift(shift)[shift_column])
        tmpdf[col_name] = tmpdf[col_name].round(3)
        
        print_log(f"price_velocity : {col_name} created")
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=[col_name])
        max_date,min_date = self.get_min_max_date(tmpdf)
        #print(tmpdf.head())
        #print(tmpdf[col_name].value_counts())
        self.create_column_and_save_to_table(time_stamp_col='timestamp', data=tmpdf)
        self.save_feature_info(function_name='price_velocity',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
        if return_df:
            return tmpdf
        del tmpdf

    def price_velocity_rate(self, func_dict_args, tmpdf=None, return_df=False):
        # freq='D',shift=5,shift_column=['close','open']
        print_log("*"*100)
        print_log(f"price_velocity called with arguments {func_dict_args}")
        freq = func_dict_args.get('freq')
        shift = func_dict_args.get('shift')
        shift_column = func_dict_args.get('shift_column')

        if freq is not None:
            col_name = f'ROLLPVR_{shift_column}_{freq}_{shift}'.replace(
                "-", '_minus_')
            if tmpdf is None:
                tmpdf = self.get_ohlc_df(column_name=col_name)
                if tmpdf is None:
                    return
            tmpdf[col_name] = tmpdf[shift_column].subtract(
                tmpdf.shift(shift, freq=freq)[shift_column])/shift
        else:
            col_name = f'ROLLPVR_{shift_column}_{shift}'.replace(
                "-", '_minus_')
            if tmpdf is None:
                tmpdf = self.get_ohlc_df(column_name=col_name)
                if tmpdf is None:
                    return
            tmpdf[col_name] = tmpdf[shift_column].subtract(
                tmpdf.shift(shift)[shift_column])/shift
        tmpdf[col_name] = tmpdf[col_name].round(3)
        print_log(f"price_velocity : {col_name} created")
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=[col_name])
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        self.save_feature_info(function_name='price_velocity_rate',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
        if return_df:
            return tmpdf
        del tmpdf

    def filter_data(self, func_dict_args, df=None):
        # start_date=None,end_date=None,filter_rows=None,
        print_log("*"*100)
        print_log(f"filter_data called with arguments {func_dict_args}")
        start_date = func_dict_args.get('start_date')
        end_date = func_dict_args.get('end_date')
        filter_rows = func_dict_args.get('filter_rows')
        if df is None:
            df = self.create_timestamp_index_df()
        print_log(f"Shape of dataframe before FilterData is {df.shape}")
        if (start_date != 'None' and end_date == 'None') or (start_date is not None and end_date is None):
            df = df.sort_index().loc[start_date:]
            print_log(f"Data filtered with {start_date}")
        elif (start_date == 'None' and end_date != 'None') or (start_date is None and end_date is not None):
            df = df.sort_index().loc[:end_date]
            print_log(f"Data filtered with {end_date}")
        elif (start_date != 'None' and end_date != 'None') or (start_date is not None and end_date is not None):
            df = df.sort_index().loc[start_date:end_date]
            print_log(f"Data filtered with {end_date}")
        else:
            df = df.sort_index()
            print_log(f"No filtering done")
        if filter_rows != 'None' or filter_rows is not None:
            df = df[:filter_rows]
            print_log(f"Data filtered with filter rows {filter_rows}")
        if self.verbose:
            print_log(f"Shape of dataframe after FilterData is {df.shape}")
        return df

    def zscore(self, x, window):
        r = x.rolling(window=window)
        m = r.mean().shift(1)
        s = r.std(ddof=0).shift(1)
        z = (x-m)/s
        return z

    def rolling_zscore(self, func_dict_args, tmpdf=None, return_df=False):
        # rolling_zscore_args = {'column_name': col, 'window': wndw}
        print_log("*"*100)
        print_log(f"rolling_zscore called with arguments {func_dict_args}")
        column = func_dict_args.get('column_name')
        window = func_dict_args.get('window')
        col_name = f'ROLLZSR_{column}_{window}'.replace("-", '_minus_')
        #if tmpdf is None:
        tmpdf = self.get_ohlc_df_v2(tmpdf,column_name=col_name)
        if tmpdf is None:
            return
        merge_dict = {}
        
        merge_dict.update({col_name: self.zscore(tmpdf[column], window)})
        tmpdf = pd.concat([tmpdf, pd.concat(merge_dict, axis=1)], axis=1)
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=[col_name])
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        self.save_feature_info(function_name='rolling_zscore',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
        if return_df:
            return tmpdf
        del tmpdf

    def rolling_log_transform(self, func_dict_args, tmpdf=None, return_df=False):
        # column
        print_log("*"*100)
        print_log(f"rolling_log_transform called with arguments {func_dict_args}")
        column = func_dict_args.get('column_name')
        col_name = f'ROLLLOG_{column}'.replace("-", '_minus_')
        #if tmpdf is None:
        tmpdf = self.get_ohlc_df_v2(tmpdf,column_name=col_name)
        if tmpdf is None:
            return
        
        merge_dict = {}
        merge_dict.update({col_name: tmpdf[column].apply(np.log)})
        tmpdf = pd.concat([tmpdf, pd.concat(merge_dict, axis=1)], axis=1)
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=[col_name])
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        self.save_feature_info(function_name='rolling_log_transform',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
        if return_df:
            return tmpdf
        del tmpdf, merge_dict

    def rolling_percentage_change(self, func_dict_args, tmpdf=None, return_df=False):
        # columns,periods=30, fill_method='pad', limit=None, freq=None,verbose=False
        print_log("*"*100)
        print_log(
            f"rolling_percentage_change called with arguments {func_dict_args}")

        column = func_dict_args.get('column_name')
        periods = func_dict_args.get('periods')
        fill_method = func_dict_args.get('fill_method')
        limit = func_dict_args.get('limit')
        freq = func_dict_args.get('freq')
        col_name = f"ROLLPCH_{column}_{periods}_{freq}".replace('-','_minus_')
        if tmpdf is None:
            tmpdf = self.get_ohlc_df(column_name=col_name)
            if tmpdf is None:
                return
        merge_dict = {}
        merge_dict.update({col_name: tmpdf[column].pct_change(
            periods=periods, fill_method=fill_method, limit=limit, freq=freq)})
        tmpdf = pd.concat([tmpdf, pd.concat(merge_dict, axis=1)], axis=1)
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=[col_name])
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        self.save_feature_info(function_name='rolling_percentage_change',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
        if return_df:
            return tmpdf
        del tmpdf, merge_dict

    def rolling_percentage_change_multiplier(self, func_dict_args, tmpdf=None, return_df=False):
        # rolling_percentage_change_multiplier_args = {
        #            'column_name': col, 'periods': per, 'fill_method': 'pad', 'limit': None, 'freq': None, 'multiplier': 100}
        print_log("*"*100)
        print_log(
            f"rolling_percentage_change_multiplier called with arguments {func_dict_args}")

        column = func_dict_args.get('column_name')
        periods = func_dict_args.get('periods')
        fill_method = func_dict_args.get('fill_method')
        limit = func_dict_args.get('limit')
        freq = func_dict_args.get('freq')
        multiplier = func_dict_args.get('multiplier')
        col_name = f"ROLLPCM_{column}_{periods}_{freq}".replace('-','_minus_')
        #if tmpdf is None:
        tmpdf = self.get_ohlc_df_v2(tmpdf,column_name=col_name)
        if tmpdf is None:
            return
        #print(column)
        #print(tmpdf.head())
        merge_dict = {}
        merge_dict.update({col_name: tmpdf[column].pct_change(
            periods=periods, fill_method=fill_method, limit=limit, freq=freq)*multiplier})
        tmpdf = pd.concat([tmpdf, pd.concat(merge_dict, axis=1)], axis=1)
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=[col_name])
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        self.save_feature_info(function_name='rolling_percentage_change_multiplier',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
        if return_df:
            return tmpdf
        del tmpdf, merge_dict

    def rolling_weighted_exponential_average(self, func_dict_args, tmpdf=None, return_df=False):
        # ling_weighted_exponential_average_args = {'column_name': col, 'com': None, 'span': spn, 'halflife': None,
        #                          'alpha': None, 'min_periods': 0, 'adjust': True, 'ignore_na': False, 'axis': 0, 'times': None}
        print_log("*"*100)
        print_log(
            f"rolling_weighted_exponential_average called with arguments {func_dict_args}")

        column = func_dict_args.get('column_name')
        com = func_dict_args.get('com')
        span = func_dict_args.get('span')
        halflife = func_dict_args.get('halflife')
        alpha = func_dict_args.get('alpha')
        min_periods = func_dict_args.get('min_periods')
        adjust = func_dict_args.get('adjust')
        ignore_na = func_dict_args.get('ignore_na')
        axis = func_dict_args.get('axis')
        times = func_dict_args.get('times')
        col_name = f"ROLLWEA_{column}_{span}".replace('-','_minus_')
        if tmpdf is None:
            tmpdf = self.get_ohlc_df(column_name=col_name)
            if tmpdf is None:
                return
        merge_dict = {}
        merge_dict.update({col_name: tmpdf[column].ewm(
            com=com, span=span, halflife=halflife, alpha=alpha, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na, axis=axis, times=times).mean()})
        tmpdf = pd.concat([tmpdf, pd.concat(merge_dict, axis=1)], axis=1)
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=[col_name])
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        self.save_feature_info(function_name='rolling_weighted_exponential_average',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
        if return_df:
            return tmpdf
        del tmpdf, merge_dict

    def rolling_percentile(self, func_dict_args, tmpdf=None, return_df=False):
        # rolling_percentile_args = {'column_name': col, 'window': window, 'min_periods': None, 'quantile': None}
        print_log("*"*100)
        print_log(f"rolling_percentile called with arguments {func_dict_args}")

        column = func_dict_args.get('column_name')
        window = func_dict_args.get('window')
        min_periods = func_dict_args.get('min_periods')
        quantile = func_dict_args.get('quantile')
        col_name = f"ROLLPCT_{column}_{window}_{min_periods}_{quantile}".replace('-','_minus_').replace('.','')
        if tmpdf is None:
            tmpdf = self.get_ohlc_df(column_name=col_name)
            if tmpdf is None:
                return
        merge_dict = {}
        merge_dict.update({col_name: tmpdf[column].rolling(
            window, min_periods=min_periods).apply(lambda x: np.array(x)[-1] - np.quantile(np.array(x), q=quantile))})
        tmpdf = pd.concat([tmpdf, pd.concat(merge_dict, axis=1)], axis=1)
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=[col_name])
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        self.save_feature_info(function_name='rolling_percentile',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
        if return_df:
            return tmpdf
        del tmpdf, merge_dict

    def rank(self, s):
        # s = pd.Series(array)
        return s.rank(ascending=False)[len(s)-1]

    def rolling_rank(self, func_dict_args, tmpdf=None, return_df=False):
        # rolling_rank_args = {'column_name': col, 'window': window, 'min_periods': None}
        print_log("*"*100)
        print_log(f"rolling_rank called with arguments {func_dict_args}")

        column = func_dict_args.get('column_name')
        window = func_dict_args.get('window')
        min_periods = func_dict_args.get('min_periods')
        col_name = f"ROLLRNK_{column}_{window}_{min_periods}".replace('-','_minus_')
        #if tmpdf is None:
        #   tmpdf = self.get_ohlc_df_v2(column_name=col_name)
        tmpdf = self.get_ohlc_df_v2(tmpdf,column_name=col_name)
        if tmpdf is None:
            return
        merge_dict = {}
        merge_dict.update({col_name: tmpdf[column].rolling(
            window=window, min_periods=min_periods).apply(self.rank)})
        tmpdf = pd.concat([tmpdf, pd.concat(merge_dict, axis=1)], axis=1)
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=[col_name])
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        self.save_feature_info(function_name='rolling_rank',feature_name=col_name,max_date=max_date,
                               min_date=min_date,status='saved',feature_args=func_dict_args)
        if return_df:
            return tmpdf
        del tmpdf, merge_dict

    def rolling_binning(self, func_dict_args, tmpdf=None, return_df=False):
        # rolling_binning_args = {'column_name': col, 'window': window, 'min_periods': None,'get_current_row_bin':True,'n_bins':n_bin}
        print_log("*"*100)
        print_log(f"rolling_binning called with arguments {func_dict_args}")

        column = func_dict_args.get('column_name')
        window = func_dict_args.get('window')
        min_periods = func_dict_args.get('min_periods')
        get_current_row_bin = func_dict_args.get('get_current_row_bin')
        n_bins = func_dict_args.get('n_bins')
        col_name = f"ROLLBIN_{column}_{window}_{min_periods}_{n_bins}".replace('-','_minus_')
        #if tmpdf is None:
        #    tmpdf = self.get_ohlc_df_v2(column_name=col_name)
        tmpdf = self.get_ohlc_df_v2(tmpdf,column_name=col_name)
        if tmpdf is None:
            return
        if get_current_row_bin:
            def bin_roll_fxn(x):
                return pd.qcut(np.array(x), labels=False, q=n_bins, duplicates='drop')[-1]
        else:
            def bin_roll_fxn(x):
                return pd.qcut(np.array(x), labels=False, q=n_bins, duplicates='drop')[0]
        merge_dict = {}
        merge_dict.update({col_name: tmpdf[column].rolling(
            window=window, min_periods=min_periods).apply(bin_roll_fxn)})
        tmpdf = pd.concat([tmpdf, pd.concat(merge_dict, axis=1)], axis=1)
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=[col_name])
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        self.save_feature_info(function_name='rolling_binning',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
        if return_df:
            return tmpdf
        del tmpdf, merge_dict

    def rolling_trends(self, func_dict_args, tmpdf=None, return_df=False):
        # rolling_trends_args = {'column_name': col, 'window': window, 'min_periods': None}
        print_log("*"*100)
        print_log(f"rolling_trends called with arguments {func_dict_args}")

        column = func_dict_args.get('column_name')
        window = func_dict_args.get('window')
        min_periods = func_dict_args.get('min_periods')
        col_name = f"ROLLTRN_{column}_{window}_{min_periods}".replace('-','_minus_')
        if tmpdf is None:
            tmpdf = self.get_ohlc_df(column_name=col_name)
            if tmpdf is None:
                return

        merge_dict = {}
        merge_dict.update({col_name: tmpdf[column].pct_change(
        ).apply(np.sign).rolling(window, min_periods=min_periods).apply(np.sum)})
        tmpdf = pd.concat([tmpdf, pd.concat(merge_dict, axis=1)], axis=1)
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=[col_name])
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        self.save_feature_info(function_name='rolling_trends',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
        if return_df:
            return tmpdf
        del tmpdf, merge_dict

    def column_diff_on_basis_col(self, func_dict_args, tmpdf=None, return_df=False):
        # column_diff_on_basis_col_args = {'column_name': col, 'basis_column': 'open'}
        print_log("*"*100)
        print_log(f"column_diff_on_basis_col called with arguments {func_dict_args}")

        column = func_dict_args.get('column_name')
        basis_column = func_dict_args.get('basis_column')
        col_name = f"DIFFCOL_{column}_{basis_column}".replace('-','_minus_')
        #if tmpdf is None:
        #   tmpdf = self.get_ohlc_df(column_name=col_name)
        tmpdf = self.get_ohlc_df_v2(tmpdf,column_name=col_name)
        if tmpdf is None:
            return
        #print(tmpdf.head())
        #print("------")
        merge_dict = {}

        merge_dict.update({col_name: tmpdf[basis_column] - tmpdf[column]})
        print_log(f"{col_name} completed")
        tmpdf = pd.concat([tmpdf,pd.concat(merge_dict,axis=1)],axis=1)
        
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        #print(tmpdf.head())
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=[col_name])
        max_date,min_date = self.get_min_max_date(tmpdf)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        self.save_feature_info(function_name='column_diff_on_basis_col',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',
                               feature_args=func_dict_args)
        if return_df:
            return tmpdf
        del tmpdf, merge_dict
        
    def rolling_stats(self, func_dict_args, tmpdf=None, return_df=False):
        # rolling_stats_args = {'column_name': col, 'window': window, 'min_periods': None}
        print_log("*"*100)
        print_log(f"rolling_stats called with arguments {func_dict_args}")

        column = func_dict_args.get('column_name')
        window = func_dict_args.get('window')
        min_periods = func_dict_args.get('min_periods')
        col_name = f"ROLLSTT_{column}_{window}_{min_periods}_MAXMIN".replace('-','_minus_')
        if tmpdf is None:
            tmpdf = self.get_ohlc_df(column_name=col_name)
            if tmpdf is None:
                return
        max_date,min_date = self.get_min_max_date(tmpdf)
        column_list = []
        merge_dict = {}
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_diff':
            func_dict_args.update({'lookback_func':'lookback_diff'})
            col_name = f"ROLLSTT_{column}_{window}_{min_periods}_DIFF".replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(
                window, min_periods=min_periods).apply(lambda x: np.array(x)[-1]-np.array(x)[0])})
            column_list.append(col_name)
            print_log(f"ROLLSTT_{column}_{window}_{min_periods}_DIFF completed")
            self.save_feature_info(function_name='rolling_stats',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)

        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_maxdiff':
            func_dict_args.update({'lookback_func':'lookback_maxdiff'})       
            col_name = f"ROLLSTT_{column}_{window}_{min_periods}_MAXDIFF".replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(
                window, min_periods=min_periods).apply(lambda x: np.array(x)[-1]-max(np.array(x)))})
            column_list.append(col_name)
            self.save_feature_info(function_name='rolling_stats',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            print_log(f"{col_name} completed")

        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_mindiff':
            func_dict_args.update({'lookback_func':'lookback_mindiff'})        
            col_name = f"ROLLSTT_{column}_{window}_{min_periods}_MINDIFF".replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(
                window, min_periods=min_periods).apply(lambda x: np.array(x)[-1]-min(np.array(x)))})
            column_list.append(col_name)
            self.save_feature_info(function_name='rolling_stats',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            print_log(f"{col_name} completed")
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_meandiff':
            func_dict_args.update({'lookback_func':'lookback_meandiff'})     
            col_name = f"ROLLSTT_{column}_{window}_{min_periods}_MEANDIFF".replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(
                window, min_periods=min_periods).apply(lambda x: np.array(x)[-1]-mean(np.array(x)))})
            column_list.append(col_name)
            self.save_feature_info(function_name='rolling_stats',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            print_log(f"{col_name} completed")
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_maxmin':
            func_dict_args.update({'lookback_func':'lookback_maxmin'})   
            col_name = f"ROLLSTT_{column}_{window}_{min_periods}_MAXMIN".replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(
                window, min_periods=min_periods).apply(lambda x: max(np.array(x))-min(np.array(x)))})
            column_list.append(col_name)
            self.save_feature_info(function_name='rolling_stats',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            print_log(f"{col_name} completed")
        
        #column_list.append('timestamp')
        tmpdf = pd.concat([tmpdf, pd.concat(merge_dict, axis=1)], axis=1)
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=column_list)
        #tmpdf = tmpdf[column_list]
        
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        if return_df:
            return tmpdf
        del tmpdf, merge_dict

    def rolling_stats_lookback(self, func_dict_args, tmpdf=None, return_df=False):
        # columns,window=60,lookback_divider =2,min_periods=None
        print_log("*"*100)
        print_log(
            f"rolling_stats_lookback called with arguments {func_dict_args}")

        column = func_dict_args.get('column_name')
        window = func_dict_args.get('window')
        min_periods = func_dict_args.get('min_periods')
        self.lookback_divider = func_dict_args.get('lookback_divider')

        def lookback_diff(vals):
            offset = len(vals)//self.lookback_divider
            return np.array(vals)[offset]-np.array(vals)[0]

        def lookback_max(vals):
            offset = len(vals)//self.lookback_divider
            return np.array(vals)[offset]-max(np.array(vals)[0:offset+1])

        def lookback_min(vals):
            offset = len(vals)//self.lookback_divider
            return np.array(vals)[offset]-min(np.array(vals)[0:offset+1])

        def lookback_mean(vals):
            offset = len(vals)//self.lookback_divider
            return np.array(vals)[offset]-np.array(vals)[0:offset+1].mean()

        def lookback_max_max(vals):
            offset = len(vals)//self.lookback_divider
            return max(np.array(vals))-np.array(vals)[0:offset+1].mean()

        def lookback_min_min(vals):
            offset = len(vals)//self.lookback_divider
            return min(np.array(vals))-np.array(vals)[0:offset+1].mean()
        column_list = []
        col_name = f'ROLLSLB_{column}_{window}_{min_periods}_{self.lookback_divider}_MINMIN'.replace('-','_minus_')
        if tmpdf is None:
            tmpdf = self.get_ohlc_df(column_name=col_name)
            if tmpdf is None:
                return
        #print("rolling_stats_lookback------------")
        #print(tmpdf.head())
        max_date,min_date = self.get_min_max_date(tmpdf)
        merge_dict = {}
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_diff':
            func_dict_args.update({'lookback_func':'lookback_diff'})
            col_name = f'ROLLSLB_{column}_{window}_{min_periods}_{self.lookback_divider}_DIFF'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_diff)})
            self.save_feature_info(function_name='rolling_stats_lookback',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_max':
            func_dict_args.update({'lookback_func':'lookback_max'})
            col_name = f'ROLLSLB_{column}_{window}_{min_periods}_{self.lookback_divider}_MAXDIFF'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_max)})
            self.save_feature_info(function_name='rolling_stats_lookback',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_min':
            func_dict_args.update({'lookback_func':'lookback_min'})
            col_name = f'ROLLSLB_{column}_{window}_{min_periods}_{self.lookback_divider}_MINDIFF'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_min)})
            self.save_feature_info(function_name='rolling_stats_lookback',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")

        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_mean':
            func_dict_args.update({'lookback_func':'lookback_mean'})
            col_name = f'ROLLSLB_{column}_{window}_{min_periods}_{self.lookback_divider}_MEANDIFF'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_mean)})
            self.save_feature_info(function_name='rolling_stats_lookback',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")

        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_max_max':
            func_dict_args.update({'lookback_func':'lookback_max_max'})       
            col_name = f'ROLLSLB_{column}_{window}_{min_periods}_{self.lookback_divider}_MAXMAX'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_max_max)})
            self.save_feature_info(function_name='rolling_stats_lookback',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")

        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_min_min':
            func_dict_args.update({'lookback_func':'lookback_min_min'})            
            col_name = f'ROLLSLB_{column}_{window}_{min_periods}_{self.lookback_divider}_MINMIN'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_min_min)})
            self.save_feature_info(function_name='rolling_stats_lookback',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
        
        tmpdf = pd.concat([tmpdf, pd.concat(merge_dict, axis=1)], axis=1)
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        #print(tmpdf.head())
        #print(column_list)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=column_list)
        self.create_column_and_save_to_table(time_stamp_col='timestamp', data=tmpdf)
        if return_df:
            return tmpdf
        del tmpdf, merge_dict

    def rolling_stats_lookback_compare(self, func_dict_args, tmpdf=None, return_df=False):
        # rolling_stats_lookback_compare_args = {'column_name': col, 'window': window, 'min_periods': None,'lookback_divider':2}
        print_log("*"*100)
        print_log(
            f"rolling_stats_lookback_compare called with arguments {func_dict_args}")

        column = func_dict_args.get('column_name')
        window = func_dict_args.get('window')
        min_periods = func_dict_args.get('min_periods')
        self.lookback_divider = func_dict_args.get('lookback_divider')

        def lookback_diff(vals):
            offset = len(vals)//self.lookback_divider
            res = (np.array(vals)[offset]-np.array(vals)[-1]
                   ) - (np.array(vals)[0]-np.array(vals)[offset])
            return res

        def lookback_max(vals):
            offset = len(vals)//self.lookback_divider
            return max(np.array(vals)[offset+1:])-max(np.array(vals)[0:offset+1])

        def lookback_min(vals):
            offset = len(vals)//self.lookback_divider
            return min(np.array(vals)[offset+1:])-min(np.array(vals)[0:offset+1])

        def lookback_mean(vals):
            offset = len(vals)//self.lookback_divider
            return np.array(vals)[offset+1:].mean()-np.array(vals)[0:offset+1].mean()

        def lookback_max_min(vals):
            offset = len(vals)//self.lookback_divider
            return max(np.array(vals)[offset+1:])-min(np.array(vals)[0:offset+1])

        def lookback_min_max(vals):
            offset = len(vals)//self.lookback_divider
            return min(np.array(vals)[offset+1:])-max(np.array(vals)[0:offset+1])

        def lookback_sum(vals):
            offset = len(vals)//self.lookback_divider
            return sum(np.array(vals)[offset+1:])-sum(np.array(vals)[0:offset+1])
                
        col_name = f'ROLLLBC_{column}_{window}_{min_periods}_{self.lookback_divider}_MINMAX'.replace('-','_minus_')
        
        if tmpdf is None:
            tmpdf = self.get_ohlc_df(column_name=col_name)
            if tmpdf is None:
                return
        max_date,min_date = self.get_min_max_date(tmpdf)
        merge_dict = {}
        column_list = []
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_diff':
            func_dict_args.update({'lookback_func':'lookback_diff'})
            col_name = f'ROLLLBC_{column}_{window}_{min_periods}_{self.lookback_divider}_DIFF'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_diff)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_max':
            func_dict_args.update({'lookback_func':'lookback_max'})
            col_name = f'ROLLLBC_{column}_{window}_{min_periods}_{self.lookback_divider}_MAXDIFF'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_max)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_min':
            func_dict_args.update({'lookback_func':'lookback_min'})
            col_name = f'ROLLLBC_{column}_{window}_{min_periods}_{self.lookback_divider}_MINDIFF'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_min)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_mean':
            func_dict_args.update({'lookback_func':'lookback_mean'})
            col_name = f'ROLLLBC_{column}_{window}_{min_periods}_{self.lookback_divider}_MEANDIFF'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_mean)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_max_min':
            func_dict_args.update({'lookback_func':'lookback_max_min'})
            col_name = f'ROLLLBC_{column}_{window}_{min_periods}_{self.lookback_divider}_MAXMIN'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_max_min)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_min_max':
            func_dict_args.update({'lookback_func':'lookback_min_max'})
            col_name = f'ROLLLBC_{column}_{window}_{min_periods}_{self.lookback_divider}_MINMAX'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_min_max)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
        
        tmpdf = pd.concat([tmpdf, pd.concat(merge_dict, axis=1)], axis=1)
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=column_list)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        if return_df:
            return tmpdf
        del tmpdf, merge_dict

    def rolling_stats_lookback_compare_offset(self, func_dict_args, tmpdf=None, return_df=False):
        # rolling_stats_lookback_compare_args = {'column_name': col, 'window': window, 'min_periods': None,'lookback_divider':2}
        print_log("*"*100)
        print_log(
            f"rolling_stats_lookback_compare_offset called with arguments {func_dict_args}")

        column = func_dict_args.get('column_name')
        window = func_dict_args.get('window')
        min_periods = func_dict_args.get('min_periods')
        self.offset = func_dict_args.get('offset')
        self.lookback_divider = func_dict_args.get('lookback_divider')

        def lookback_diff(vals):
            res = (np.array(vals)[self.offset]-np.array(vals)[-1]
                   ) - (np.array(vals)[0]-np.array(vals)[self.offset])
            return res

        def lookback_max(vals):
            return max(np.array(vals)[self.offset+1:])-max(np.array(vals))

        def lookback_min(vals):
            return min(np.array(vals)[self.offset+1:])-min(np.array(vals))

        def lookback_mean(vals):
            return np.array(vals)[self.offset+1:].mean()-np.array(vals).mean()

        def lookback_max_min(vals):
            return max(np.array(vals)[self.offset+1:])-min(np.array(vals))

        def lookback_min_max(vals):
            return min(np.array(vals)[self.offset+1:])-max(np.array(vals))

        def lookback_sum(vals):
            return sum(np.array(vals)[self.offset+1:])-sum(np.array(vals))
                
        col_name = f'ROLLOFF_{column}_{window}_{min_periods}_{self.lookback_divider}_MINMAX'.replace('-','_minus_')
        
        if tmpdf is None:
            tmpdf = self.get_ohlc_df(column_name=col_name)
            if tmpdf is None:
                return
        max_date,min_date = self.get_min_max_date(tmpdf)
        merge_dict = {}
        column_list = []
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_diff':
            func_dict_args.update({'lookback_func':'lookback_diff'})
            col_name = f'ROLLOFF_{column}_{window}_{min_periods}_{self.lookback_divider}_DIFF'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_diff)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare_offset',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_max':
            func_dict_args.update({'lookback_func':'lookback_max'})
            col_name = f'ROLLOFF_{column}_{window}_{min_periods}_{self.lookback_divider}_MAXDIFF'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_max)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare_offset',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_min':
            func_dict_args.update({'lookback_func':'lookback_min'})
            col_name = f'ROLLOFF_{column}_{window}_{min_periods}_{self.lookback_divider}_MINDIFF'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_min)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare_offset',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_mean':
            func_dict_args.update({'lookback_func':'lookback_mean'})
            col_name = f'ROLLOFF_{column}_{window}_{min_periods}_{self.lookback_divider}_MEANDIFF'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_mean)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare_offset',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_max_min':
            func_dict_args.update({'lookback_func':'lookback_max_min'})
            col_name = f'ROLLOFF_{column}_{window}_{min_periods}_{self.lookback_divider}_MAXMIN'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_max_min)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare_offset',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_min_max':
            func_dict_args.update({'lookback_func':'lookback_min_max'})
            col_name = f'ROLLOFF_{column}_{window}_{min_periods}_{self.lookback_divider}_MINMAX'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_min_max)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare_offset',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
 
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_sum':
            func_dict_args.update({'lookback_func':'lookback_sum'})
            col_name = f'ROLLOFF_{column}_{window}_{min_periods}_{self.lookback_divider}_SUM'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_sum)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare_offset',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
                   
        tmpdf = pd.concat([tmpdf, pd.concat(merge_dict, axis=1)], axis=1)
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=column_list)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        if return_df:
            return tmpdf
        del tmpdf, merge_dict

    def rolling_stats_lookback_compare_offset_two_col(self, func_dict_args, tmpdf=None, return_df=False):
        # rolling_stats_lookback_compare_args = {'column_name': col, 'window': window, 'min_periods': None,'lookback_divider':2}
        print_log("*"*100)
        print_log(
            f"rolling_stats_lookback_compare called with arguments {func_dict_args}")

        column1 = func_dict_args.get('column_name1')
        column2 = func_dict_args.get('column_name2')
        window = func_dict_args.get('window')
        min_periods = func_dict_args.get('min_periods')
        self.offset = func_dict_args.get('offset')
        self.lookback_divider = func_dict_args.get('lookback_divider')

        def lookback_diff(vals):
            res = (np.array(vals)[self.offset]-np.array(vals)[-1]
                   ) - (np.array(vals)[0]-np.array(vals)[self.offset])
            return res

        def lookback_max(vals):
            return max(np.array(vals)[self.offset+1:])-max(np.array(vals))

        def lookback_min(vals):
            return min(np.array(vals)[self.offset+1:])-min(np.array(vals))

        def lookback_mean(vals):
            return np.array(vals)[self.offset+1:].mean()-np.array(vals).mean()

        def lookback_max_min(vals):
            return max(np.array(vals)[self.offset+1:])-min(np.array(vals))

        def lookback_min_max(vals):
            return min(np.array(vals)[self.offset+1:])-max(np.array(vals))

        def lookback_sum(vals):
            return sum(np.array(vals)[self.offset+1:])-sum(np.array(vals))
                
        col_name = f'ROLLLBC_{column}_{window}_{min_periods}_{self.lookback_divider}_MINMAX'.replace('-','_minus_')
        
        if tmpdf is None:
            tmpdf = self.get_ohlc_df(column_name=col_name)
            if tmpdf is None:
                return
        max_date,min_date = self.get_min_max_date(tmpdf)
        merge_dict = {}
        column_list = []
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_diff':
            func_dict_args.update({'lookback_func':'lookback_diff'})
            col_name = f'ROLLOFF_{column1}_{column2}_{window}_{min_periods}_{self.lookback_divider}_DIFF'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[[column1,column2]].rolling(window, min_periods=min_periods).apply(lookback_diff)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_max':
            func_dict_args.update({'lookback_func':'lookback_max'})
            col_name = f'ROLLOFF_{column}_{window}_{min_periods}_{self.lookback_divider}_MAXDIFF'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_max)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_min':
            func_dict_args.update({'lookback_func':'lookback_min'})
            col_name = f'ROLLOFF_{column}_{window}_{min_periods}_{self.lookback_divider}_MINDIFF'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_min)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_mean':
            func_dict_args.update({'lookback_func':'lookback_mean'})
            col_name = f'ROLLOFF_{column}_{window}_{min_periods}_{self.lookback_divider}_MEANDIFF'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_mean)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_max_min':
            func_dict_args.update({'lookback_func':'lookback_max_min'})
            col_name = f'ROLLOFF_{column}_{window}_{min_periods}_{self.lookback_divider}_MAXMIN'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_max_min)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
        
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_min_max':
            func_dict_args.update({'lookback_func':'lookback_min_max'})
            col_name = f'ROLLOFF_{column}_{window}_{min_periods}_{self.lookback_divider}_MINMAX'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_min_max)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
 
        if func_dict_args.get('lookback_func') is None or func_dict_args.get('lookback_func') == 'lookback_sum':
            func_dict_args.update({'lookback_func':'lookback_sum'})
            col_name = f'ROLLOFF_{column}_{window}_{min_periods}_{self.lookback_divider}_SUM'.replace('-','_minus_')
            merge_dict.update({col_name: tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_sum)})
            self.save_feature_info(function_name='rolling_stats_lookback_compare',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
            column_list.append(col_name)
            print_log(f"{col_name} completed")
                   
        tmpdf = pd.concat([tmpdf, pd.concat(merge_dict, axis=1)], axis=1)
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=column_list)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        if return_df:
            return tmpdf
        del tmpdf, merge_dict

        
    def rolling_previous_day_range(self, func_dict_args, tmpdf=None, return_df=False):
        # rolling_previous_day_range_args = {'column_name': col, 'freq': freq, 'shift_val': 1,'resample':resample}
        print_log("*"*100)
        print_log(
            f"rolling_previous_day_range called with arguments {func_dict_args}")

        column = func_dict_args.get('column_name')
        freq = func_dict_args.get('freq')
        shift_val = func_dict_args.get('shift_val')
        rsmpl = func_dict_args.get('resample')
        
        col_name = f'ROLLPVR_{column}_{freq}_{shift_val}_{rsmpl}'.replace('-','_minus_')
        if tmpdf is None:
            tmpdf = self.get_ohlc_df(column_name=col_name)
            if tmpdf is None:
                return
        max_date,min_date = self.get_min_max_date(tmpdf)
        merge_dict = {}
        column_list = []
        
        merge_dict.update({col_name: tmpdf[column].resample(rsmpl).ffill(
        ).groupby(pd.Grouper(freq=freq)).apply(lambda x: np.array(x)[-1]-np.array(x)[0]).shift(shift_val)})
        self.save_feature_info(function_name='rolling_previous_day_range',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
        column_list.append(col_name)
        print_log(f"{col_name} completed")
        
        col_name = f'ROLLPVR_{column}_{freq}_{shift_val}_{rsmpl}_MAX_MIN'.replace('-','_minus_')
        merge_dict.update({col_name: tmpdf[column].resample(
            rsmpl).ffill().groupby(pd.Grouper(freq=freq)).apply(lambda x: max(np.array(x))-min(np.array(x))).shift(shift_val)})
        
        self.save_feature_info(function_name='rolling_previous_day_range',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)
        column_list.append(col_name)
        print_log(f"{col_name} completed")
        
        tmpdf = pd.merge_asof(tmpdf, pd.concat(merge_dict, axis=1),left_index=True, right_index=True)
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=column_list)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        if return_df:
            return tmpdf
        del tmpdf, merge_dict

    def rolling_gap_open_min(self, func_dict_args, tmpdf=None, return_df=False):
        # rolling_gap_open_min_args = {'column_name': col}
        print_log("*"*100)
        print_log(
            f"rolling_previous_day_range called with arguments {func_dict_args}")

        column = func_dict_args.get('column_name')
        col_name = f'ROLLGOM_{column}'.replace('-','_minus_')
        if tmpdf is None:
            tmpdf = self.get_ohlc_df(column_name=col_name)
            if tmpdf is None:
                return

        merge_dict = {}
        column_list = []
        max_date,min_date = self.get_min_max_date(tmpdf)
        column_list.append(col_name)
        a = tmpdf[column].resample('d').bfill().groupby(pd.Grouper(freq='d')).apply(lambda x: x[0]).subtract(
            tmpdf[column].resample('d').ffill().groupby(pd.Grouper(freq='d')).apply(lambda x: x[-1]).fillna(0))
        merge_dict.update({col_name: a[1:]})

        tmpdf = pd.merge_asof(tmpdf, pd.concat(
            merge_dict, axis=1), left_index=True, right_index=True)
        print_log(f"{col_name} completed")
        #tmpdf = self.remove_ohlc_cols(tmpdf)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf, cols=column_list)
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=tmpdf)
        self.save_feature_info(function_name='rolling_gap_open_min',feature_name=col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)

        if return_df:
            return tmpdf
        del tmpdf, merge_dict


    def compare_two_diff(self,dist1,dist2,tolerance=2):
        val = np.abs(((np.median(dist1) - np.median(dist2))/np.median(dist1))*100)
        if val > tolerance:
            return 'differ'
        else:
            return 'same'

    def update_unstable_columns(self,tolerance,filter_ind='time_split_train_fold_10'):
        sql = f"select * from {self.train_feature_table} where time_split = '{filter_ind}'"
        df_features = ddu.load_table_df(self.db_conection,table_name=None,column_names=None,filter=None,load_sql=sql)
        df_features = df_features.loc[:,~df_features.columns.duplicated()].copy()

        dist1 = df_features['open']
        list_val = []
        for col in df_features.columns.tolist():
            if df_features[col].dtype != '<M8[ns]' and df_features[col].dtype != 'object':
                #print(col)
                dist2 = df_features[col]
                val = self.compare_two_diff(dist1,dist2,tolerance=tolerance)
                if val == 'same':
                    list_val.append(col)
        #ull_cols = du.nullcolumns(df_features)
        del df_features
        func_dict_args = {'filter_ind':filter_ind,'tolerance':tolerance}
        for col in list_val:
            self.save_feature_info(function_name='update_unstable_columns',feature_name=col,column_type='unstable_column',max_date='1999-01-01 09:00:00',min_date='1999-01-01 09:00:00',status='saved',feature_args=func_dict_args)
        #return list_val
    
    def save_check_point(self,end_conection=False):
        if self.database_path is not None:
            self.db_conection.close()
            if not end_conection:
                du.print_log(f"Checkpointing successful")
                self.db_conection = duckdb.connect(database=self.database_path , read_only=False)
            else:
                du.print_log(f"Connection closed")
        else:
            du.print_log(f"Checkpointing not possible as database_path is {self.database_path}")
                                 
    def treat_unstable_cols(self, func_dict_args):
        # treat_unstable_cols_args = {'basis_column':'close','ohlc_columns':['close','open','high','low'],'tolerance':2,'transform_options':['rank'],filter_ind='time_split_train_fold_10'}
        print_log("*"*100)
        print_log(
            f"treat_unstable_cols called with arguments {func_dict_args}")

        #df = self.get_ohlc_df()
        
        basis_column = func_dict_args.get('basis_column')
        #tolerance = func_dict_args.get('tolerance')
        transform_options = func_dict_args.get('transform_options')
        #filter_ind = func_dict_args.get('filter_ind')
        unstable_cols_dict = self.get_features_with_load_date(feature_type='unstable_column')
                
        du.print_log(f"Number of unstable column : {len(unstable_cols_dict)}")
        if len(unstable_cols_dict)>0:
            for transform_option in transform_options:
                for col ,load_date in unstable_cols_dict.items():
                    du.print_log(f"Running unstable column treatment for {col} for load date {load_date} and transform option {transform_option}")
                    #if load_date is None:
                    sql = f"select timestamp,{basis_column},{col} from {self.train_feature_table}"
                    #elif load_date <= self.max_date_raw_table:
                    #    du.print_log(f"Load date is {load_date} and max_date_raw_table is {self.max_date_raw_table}. Therefore skipping {col}")
                    #    continue
                    #else:
                     #   sql = f"select timestamp,{basis_column},{col} from {self.train_feature_table} where timestamp > {load_date}"
                    df = ddu.load_table_df(self.db_conection,table_name=None,column_names=None,filter=None,load_sql=sql)
                    print_log(f"Loaded df from sql is {df.shape}")
                    func_dict_args_tmp = {"column_name":col,"basis_column":basis_column}
                    self.column_diff_on_basis_col(func_dict_args_tmp, tmpdf=df, return_df=False)
                    if col not in self.ohlc_column:
                        if transform_option == 'bint':
                            #column = func_dict_args.get('column_name')
                            func_dict_args_tmp = {'column_name': col, 'window': 15,
                                            'min_periods': None, 'get_current_row_bin': True, 'n_bins': 5}
                            self.rolling_binning(func_dict_args_tmp, df, return_df=False)
                            func_dict_args_tmp = {'column_name': col, 'window': 30,
                                            'min_periods': None, 'get_current_row_bin': True, 'n_bins': 5}
                            self.rolling_binning(func_dict_args_tmp, df, return_df=False)
                            func_dict_args_tmp = {'column_name': col, 'window': 45,
                                            'min_periods': None, 'get_current_row_bin': True, 'n_bins': 5}
                            self.rolling_binning(func_dict_args_tmp, df, return_df=False)
                            func_dict_args_tmp = {'column_name': col, 'window': 60,
                                            'min_periods': None, 'get_current_row_bin': True, 'n_bins': 5}
                            self.rolling_binning(func_dict_args_tmp, df, return_df=False)
                        elif transform_option == 'rank':
                            func_dict_args_tmp = {'column_name': col,
                                            'window': 15, 'min_periods': None}
                            self.rolling_rank(func_dict_args_tmp, tmpdf=df,return_df=False)
                            func_dict_args_tmp = {'column_name': col,
                                            'window': 30, 'min_periods': None}
                            self.rolling_rank(func_dict_args_tmp, tmpdf=df,return_df=False)
                            func_dict_args_tmp = {'column_name': col,
                                            'window': 45, 'min_periods': None}
                            self.rolling_rank(func_dict_args_tmp, tmpdf=df,return_df=False)
                            func_dict_args_tmp = {'column_name': col,
                                            'window': 60, 'min_periods': None}
                            self.rolling_rank(func_dict_args_tmp, tmpdf=df,return_df=False)
                        elif transform_option == 'log_transform':
                            func_dict_args_tmp = {'column_name': col}
                            self.rolling_log_transform(func_dict_args_tmp, tmpdf=df,return_df=False)
                        elif transform_option == 'zscore':
                            func_dict_args_tmp = {'column_name': col, 'window': 15}
                            self.rolling_zscore(func_dict_args_tmp, tmpdf=df,return_df=False)
                            func_dict_args_tmp = {'column_name': col, 'window': 30}
                            self.rolling_zscore(func_dict_args_tmp, tmpdf=df,return_df=False)
                            func_dict_args_tmp = {'column_name': col, 'window': 45}
                            self.rolling_zscore(func_dict_args_tmp, tmpdf=df,return_df=False)
                            func_dict_args_tmp = {'column_name': col, 'window': 60}
                            self.rolling_zscore(func_dict_args_tmp, tmpdf=df,return_df=False)
                        else:
                            # columns,periods=30, fill_method='pad', limit=None, freq=None,multiplier=100
                            func_dict_args_tmp = {'column_name': col, 'periods': 15, 'fill_method': 'pad',
                                            'limit': None, 'freq': None, 'multiplier': 100}
                                    
                            self.rolling_percentage_change_multiplier(
                                func_dict_args_tmp, tmpdf=df,return_df=False)
                            func_dict_args_tmp = {'column_name': col, 'periods': 30, 'fill_method': 'pad',
                                            'limit': None, 'freq': None, 'multiplier': 100}
                            self.rolling_percentage_change_multiplier(
                                func_dict_args_tmp, tmpdf=df,return_df=False)
                            func_dict_args = {'column_name': col, 'periods': 45, 'fill_method': 'pad',
                                            'limit': None, 'freq': None, 'multiplier': 100}
                            self.rolling_percentage_change_multiplier(
                                func_dict_args_tmp, tmpdf=df,return_df=False)
                            func_dict_args_tmp = {'column_name': col, 'periods': 60, 'fill_method': 'pad',
                                            'limit': None, 'freq': None, 'multiplier': 100}
                            self.rolling_percentage_change_multiplier(
                                func_dict_args_tmp, tmpdf=df,return_df=False)
                    self.save_check_point(end_conection=False)
                    #max_date,min_date = self.get_min_max_date(df)
                    #if load_date is not None:
                    #    min_date = None
                    gc.enable()
                    del df
                    gc.collect()
                    #self.save_feature_info_v2(function_name='treat_unstable_cols',feature_name=col,column_type='unstable_column',max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args)

    def get_date_limit(self,split_type,start_date,range_picker=None,start_split=None,end_split=None):
        if range_picker is None:
            if start_split is None or end_split is None:
                start_split = 2
                end_split = 2
                split_type = 'month'
            range_picker = random.randint(start_split,end_split)
        #print(f"range_picker : {range_picker}")
        if split_type == 'date':
            date_limit = start_date + relativedelta(dt1=start_split, dt2=end_split)
        elif split_type == 'month':
            date_limit = start_date + relativedelta(months=range_picker)
            #print(f"date_limit : {date_limit}, start_date : {start_date} , delta {relativedelta(month=range_picker)}")
        elif split_type == 'days':
            date_limit = start_date + relativedelta(days=range_picker)            
        elif split_type == 'weeks':
            date_limit = start_date + relativedelta(weeks=range_picker)          
        elif split_type == 'hours':
            date_limit = start_date + relativedelta(hours=range_picker)  
        else:
            date_limit = start_date + relativedelta(months=range_picker) 
        return date_limit          
                    
    def get_split_lists(self,df,split_type,buffer_type,buffer=2,splits = {'train':[3,8],'validation':[1,3],'test':[1,3]}):
        
        min_date = df.index.min()
        max_date = df.index.max()
            
        train_range_list = []
        val_range_list = []
        test_range_list = []
        buffer1_range_list = []
        buffer2_range_list = []
        first_filter_limit = self.get_date_limit(split_type='month',start_date=min_date,range_picker=2,start_split=None,end_split=None)
        new_min_date = self.get_date_limit(split_type='month',start_date=min_date,range_picker=2,start_split=None,end_split=None)
        first_filter_range_limit = [[min_date,first_filter_limit]]
        last_filter_limit = self.get_date_limit(split_type='month',start_date=max_date,range_picker=-2,start_split=None,end_split=None)
        new_max_date = self.get_date_limit(split_type='month',start_date=max_date,range_picker=-2,start_split=None,end_split=None)

        
        while new_max_date > new_min_date:
            #train_date_limit = min_date+ relativedelta(months=train_range_picker)
            #print_log(f"Max date is {str(max_date)} and Min date is {str(min_date)}")
            train_date_limit = self.get_date_limit(split_type=split_type,start_date=new_min_date,range_picker=None,start_split=splits['train'][0],end_split=splits['train'][1])
            if train_date_limit< new_max_date:
                train_range_list.append([new_min_date,train_date_limit])
            #buffer_date_limit1 = train_date_limit + relativedelta(months=buffer)
            buffer_date_limit1 = self.get_date_limit(split_type=buffer_type,start_date=train_date_limit,range_picker=buffer,start_split=None,end_split=None)

            if buffer_date_limit1< new_max_date:
                buffer1_range_list.append([train_date_limit,buffer_date_limit1])
            #validation_date_limit = buffer_date_limit1 + relativedelta(months=validation_range_picker)
            validation_date_limit = self.get_date_limit(split_type=split_type,start_date=buffer_date_limit1,range_picker=None,start_split=splits['validation'][0],end_split=splits['validation'][1])

            if validation_date_limit< new_max_date:
                val_range_list.append([buffer_date_limit1,validation_date_limit])
            #buffer_date_limit2 = validation_date_limit + relativedelta(months=buffer)
            buffer_date_limit2 = self.get_date_limit(split_type=buffer_type,start_date=validation_date_limit,range_picker=buffer,start_split=None,end_split=None)

            if buffer_date_limit2< new_max_date:
                buffer2_range_list.append([validation_date_limit,buffer_date_limit2])
            #test_date_limit = buffer_date_limit2 + relativedelta(months=test_range_picker)
            test_date_limit = self.get_date_limit(split_type=split_type,start_date=buffer_date_limit2,range_picker=None,start_split=splits['test'][0],end_split=splits['test'][1])
            if test_date_limit< new_max_date:
                test_range_list.append([buffer_date_limit2,test_date_limit])
            new_min_date = test_date_limit
            #print_log(f"min_date : {min_date} ,max_date : {max_date} ,train_date_limit : {train_date_limit},buffer_date_limit1 : {buffer_date_limit1},validation_date_limit : {validation_date_limit},buffer_date_limit2 : {buffer_date_limit2},test_date_limit : {test_date_limit}")
        last_filter_range_limit = [[test_date_limit,max_date]]
        return first_filter_range_limit,train_range_list,buffer1_range_list,val_range_list,buffer2_range_list,test_range_list,last_filter_range_limit
            
    def time_series_column_creator(self,func_dict_args,df=None,return_df=None):
        print_log("*"*100)
        print_log(
            f"time_series_column_creator called with arguments {func_dict_args}")
        if df is None:
            df = self.get_ohlc_df()
            if df is None:
                return
        else:
            df = self.create_timestamp_index_df(df)
        #print(df.head())
        df = df.sort_index()
        print_log(f"Shape of the input dataframe is {df.shape}")
        print_log(df.dtypes)
        split_type = func_dict_args.get('split_type')
        buffer_type = func_dict_args.get('buffer_type')
        buffer = func_dict_args.get('buffer')
        splits = func_dict_args.get('splits')           
        split_col_name = func_dict_args.get('split_col_name')           
        
        if split_type is None:  
            split_type=self.train_split_type
        if buffer_type is None: 
            buffer_type=self.train_buffer_type
        if buffer is None:
            buffer=self.train_buffer
        if splits is None:
            splits=self.train_splits
        if split_col_name is None:
            split_col_name = self.train_split_col_name
                                
        first_filter_range_limit,train_range_list,buffer1_range_list,val_range_list,buffer2_range_list,test_range_list,last_filter_range_limit = self.get_split_lists(df,split_type,buffer_type,buffer=buffer,splits = splits)
        ret_df =[]
        print_log(f"first_filter_range_limit :{first_filter_range_limit},last_filter_range_limit : {last_filter_range_limit}")
        data_label = ['firstbuffer','train','buffer1','validation','buffer2','test','lastbuffer']
        cnt = 0
        for i,range_list in enumerate([first_filter_range_limit,train_range_list,buffer1_range_list,val_range_list,buffer2_range_list,test_range_list,last_filter_range_limit],1):            
            for j,time_limit in enumerate(range_list,1):
                print('*'*100)
                print(f"Time limits are {time_limit[0]}, {time_limit[1]}")
                
                tmpdf = df[df.index.to_series().between(time_limit[0], time_limit[1],inclusive='left')]
                #tmpdf[split_col_name] = f'{split_col_name}_{range_list[1]}_fold_{i}'
                tmpdf[split_col_name] = f'{split_col_name}_{data_label[i-1]}_fold_{j}'
                #print_log(f'complete for split {split_col_name}_{range_list[1]}_fold_{i}')
                print(tmpdf[['timestamp',split_col_name]])
                print_log(f'complete for split {split_col_name}_{data_label[i-1]}_fold_{j}')
                cnt = cnt + len(tmpdf)
                ret_df.append(tmpdf)
        ret_df = pd.concat(ret_df)
        ret_df = ret_df.sort_index()
        print_log(ret_df.dtypes)
        #ret_df = self.remove_ohlc_cols(ret_df)
        print(f'FINAL COUNT IS {cnt}')
        ret_df = ret_df[['timestamp',split_col_name]]
        max_date,min_date = self.get_min_max_date(ret_df)
        #print(ret_df)
        func_dict_args = {'splits':splits,'split_type':split_type,'buffer_type':buffer_type,'buffer':buffer,'split_col_name':split_col_name}
        self.create_column_and_save_to_table(
            time_stamp_col='timestamp', data=ret_df)
        self.save_feature_info(function_name='time_series_column_creator',feature_name=split_col_name,max_date=max_date,min_date=min_date,status='saved',feature_args=func_dict_args,column_type='splittercolumn')
        if return_df:
            return ret_df
        del ret_df
        
    def update_nulls(self,filter_ind,tolerance_count=6000):
        nulls_df = self.find_null_columns(filter_ind=filter_ind,tolerance_count=tolerance_count)
        func_dict_args = {"tolerance_count":tolerance_count}
        for col in nulls_df['column_name'].tolist():
            self.save_feature_info(function_name='update_nulls',
                                   feature_name=col,max_date='1999-01-01 09:00:00',min_date='1999-01-01 09:00:00',status='saved',feature_args=func_dict_args,column_type='null_reject')
        
    def find_null_columns(self,filter_ind,tolerance_count=6000):
        sql = f"select * from {self.train_feature_info_table}"
        info = ddu.load_table_df(self.db_conection,table_name=None,column_names=None,filter=None,load_sql=sql)
        features = info[(info['column_type'] == 'feature')]['feature_name'].tolist()
        features = [x.strip() for x in features]
        tmp_sql = f"select * from {self.train_feature_table} limit 10"
        dummy_df = ddu.load_table_df(self.db_conection,table_name=None,column_names=None,filter=None,load_sql=tmp_sql)
        all_columns = set(dummy_df.columns.tolist())
        del dummy_df
        reject_features = [col for col in features if col not in all_columns]
        print_log(f"Reject columns are :")
        print_log(reject_features)
        features = [col for col in features if col in all_columns]
        sql = f"select "
        sum_list = []
        for col in features:
            sum_list.append(f"sum(case when {col} is null then 1 else 0 end) {col}")

        sql = sql + ", ".join(sum_list) + f" from {self.train_feature_table} where time_split = '{filter_ind}'"
        print(sql)
        nulls_df = ddu.load_table_df(self.db_conection,table_name=None,column_names=None,filter=None,load_sql=sql)
        nulls_df = nulls_df.T
        nulls_df = nulls_df.reset_index()
        nulls_df.columns = ['column_name','null_count']
        nulls_df = nulls_df.sort_values(by='null_count',ascending=False)
        nulls_df = nulls_df[nulls_df['null_count'] > tolerance_count] 
        return nulls_df