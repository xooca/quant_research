
# -*- coding: utf-8 -*-
from pandas import concat, DataFrame
from pandas_ta import Imports
from pandas_ta.overlap import ema
from pandas_ta.utils import get_offset, verify_series, signals
import numpy as np
import pandas as pd

def rollstat(close, length=None, offset=None, **kwargs):

    """Indicator: Moving Average, Convergence/Divergence (MACD)"""
 
    # Validate arguments
    length = int(length) if length and length > 0 else 20
    
    close = verify_series(close,length)
    offset = get_offset(offset)

    if close is None: return
    _name = "ROLLSTT"
    offset = 2 if offset is not None else offset
    _props = f"{length}_{offset}"
    merge_dict = {}
    col_name = f"{_name}_{_props}_DIFF".replace('-','_minus_')
    merge_dict.update({col_name: close.rolling(length).apply(lambda x: np.array(x)[-1]-np.array(x)[0])})

    col_name = f"{_name}_{_props}_{offset}_MAXDIFF".replace('-','_minus_')
    merge_dict.update({col_name: close.rolling(length).apply(lambda x: np.array(x)[-1]-max(np.array(x)))})


    col_name = f"{_name}_{_props}_MINDIFF".replace('-','_minus_')
    merge_dict.update({col_name: close.rolling(length).apply(lambda x: np.array(x)[-1]-min(np.array(x)))})


    col_name = f"{_name}_{_props}_MEANDIFF".replace('-','_minus_')
    merge_dict.update({col_name: close.rolling(length).apply(lambda x: np.array(x)[-1]-np.mean(np.array(x)))})

    col_name = f"{_name}_{_props}_MAXMIN".replace('-','_minus_')
    merge_dict.update({col_name: close.rolling(length).apply(lambda x: max(np.array(x))-min(np.array(x)))})

    
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


rollstat.__doc__ = \
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
    fast (int): The short period. Default: 12
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

def rollstat_method(self, length=None, offset=None, **kwargs):
    close = self._get_column(kwargs.pop("close", "close"))
    result = rollstat(close=close, length=length, offset=offset, **kwargs)
    return self._post_process(result, **kwargs)