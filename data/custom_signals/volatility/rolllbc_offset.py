
# -*- coding: utf-8 -*-
from pandas import concat, DataFrame
from pandas_ta import Imports
from pandas_ta.overlap import ema
from pandas_ta.utils import get_offset, verify_series, signals
import numpy as np
import pandas as pd

def rolllbc_offset(close, length=None, offset=None, **kwargs):

    """Indicator: Moving Average, Convergence/Divergence (MACD)"""
 
    # Validate arguments
    offset_val = 2 if kwargs.get("divider_offset_val") is not None else kwargs.get("divider_offset_val")
    def lookback_diff(vals):
        res = (np.array(vals)[offset_val]-np.array(vals)[-1]
                ) - (np.array(vals)[0]-np.array(vals)[offset_val])
        return res

    def lookback_max(vals):
        return max(np.array(vals)[offset_val+1:])-max(np.array(vals))

    def lookback_min(vals):
        return min(np.array(vals)[offset_val+1:])-min(np.array(vals))

    def lookback_mean(vals):
        return np.array(vals)[offset_val+1:].mean()-np.array(vals).mean()

    def lookback_max_min(vals):
        return max(np.array(vals)[offset_val+1:])-min(np.array(vals))

    def lookback_min_max(vals):
        return min(np.array(vals)[offset_val+1:])-max(np.array(vals))

    def lookback_sum(vals):
            return sum(np.array(vals)[offset_val+1:])-sum(np.array(vals))

    length = int(length) if length and length > 0 else 20
    
    close = verify_series(close, length)
    offset = get_offset(offset)
    function_option_list = ['diff','maxdiff'] if kwargs.get('function_option') is None else kwargs.get('function_option')
    if close is None: return
    _name = "ROLLBO"
    _props = f"{length}_{offset}"
    merge_dict = {}
    for function_option in function_option_list:
        if function_option == 'diff' or function_option == 'all':
            col_name = f"{_name}_{_props}_DIFF".replace('-','_minus_')
            merge_dict.update({col_name: close.rolling(length).apply(lookback_diff)}) 

        if function_option == 'maxdiff' or function_option == 'all':
            col_name = f"{_name}_{_props}_{offset}_MAXDIFF".replace('-','_minus_')
            merge_dict.update({col_name: close.rolling(length).apply(lookback_max)})
    
        if function_option == 'mindiff' or function_option == 'all':
            col_name = f"{_name}_{_props}_MINDIFF".replace('-','_minus_')
            merge_dict.update({col_name: close.rolling(length).apply(lookback_min)})

        if function_option == 'meandiff' or function_option == 'all':
            col_name = f"{_name}_{_props}_MEANDIFF".replace('-','_minus_')
            merge_dict.update({col_name: close.rolling(length).apply(lookback_mean)})

        if function_option == 'maxmin' or function_option == 'all':
            col_name = f"{_name}_{_props}_MAXMIN".replace('-','_minus_')
            merge_dict.update({col_name: close.rolling(length).apply(lookback_max_min)})

        if function_option == 'minmax' or function_option == 'all':
            col_name = f"{_name}_{_props}_MINMAX".replace('-','_minus_')
            merge_dict.update({col_name: close.rolling(length).apply(lookback_min_max)})

        if function_option == 'sumdiff'  or function_option == 'all':
            col_name = f"{_name}_{_props}_SUMDIFF".replace('-','_minus_')
            merge_dict.update({col_name: close.rolling(length).apply(lookback_sum)})       
        
    #column_list.append('timestamp')
    df = pd.concat(merge_dict, axis=1)
    
    if offset != 0:
        df = df.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        df.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        df.fillna(method=kwargs["fill_method"], inplace=True)

    # Prepare DataFrame to return
    df.name = f"{_name}_{_props}"
    df.category = "volatility"
    return df


rolllbc_offset.__doc__ = \
"""Rolling Stats for calculation of various rolling stats

The MACD is a popular indicator to that is used to identify a security's trend.
While APO and MACD are the same calculation, MACD also returns two more series
called Signal and Histogram. The Signal is an EMA of MACD and the Histogram is
the difference of MACD and Signal.

Sources:
    https://www.tradingview.com/wiki/MACD_(Moving_Average_Convergence/Divergence)
    AS Mode: https://tr.tradingview.com/script/YFlKXHnP/

Calculation:
    Default Inputs:
        fast=12, slow=26, signal=9
    EMA = Exponential Moving Average
    MACD = EMA(close, fast) - EMA(close, slow)
    Signal = EMA(MACD, signal)
    Histogram = MACD - Signal

    if asmode:
        MACD = MACD - Signal
        Signal = EMA(MACD, signal)
        Histogram = MACD - Signal

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 20
    slow (int): The long period. Default: 26
    signal (int): The signal period. Default: 9
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    asmode (value, optional): When True, enables AS version of MACD.
        Default: False
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: macd, histogram, signal columns.
"""
# - Define a matching class method --------------------------------------------

def rolllbc_offset_method(self, length=None, offset=None, **kwargs):
    close = self._get_column(kwargs.pop("close", "close"))
    result = rolllbc_offset(close=close, length=length, offset=offset, **kwargs)
    return self._post_process(result, **kwargs)