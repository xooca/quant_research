
# -*- coding: utf-8 -*-
from pandas import concat, DataFrame
from pandas_ta import Imports
from pandas_ta.overlap import ema
from pandas_ta.utils import get_offset, verify_series, signals
import numpy as np
import pandas as pd

def diff_cols(first,second, offset_first=None, offset_second=None,**kwargs):

    """Indicator: Moving Average, Convergence/Divergence (MACD)"""
    
    # Validate arguments    
    first = verify_series(first)
    second = verify_series(second)
    

    if first is None or second is None: return
    
    offset_first = 0 if kwargs.get('offset_first') is None else kwargs.get('offset_first')
    offset_second = 0 if kwargs.get('offset_second') is None else kwargs.get('offset_second')
    offset_first = get_offset(offset_first)
    offset_second = get_offset(offset_second)
    
    _name = "ROLLDIFF"
    _props = f"{offset_first}_{offset_first}".replace("-", '_minus_')
    
    if offset_first != 0:
        first = first.shift(offset_first)
    if offset_second != 0:
        second = second.shift(offset_second)
        
    ret_value = first - second

    # Handle fills
    if "fillna" in kwargs:
        ret_value.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        ret_value.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    ret_value.name = f"{_name}_{_props}"
    ret_value.category = "statistics"
    
    return ret_value


diff_cols.__doc__ = \
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

def diff_cols_method(self, offset_first=None, offset_second=None, **kwargs):
    first = self._get_column(kwargs.pop("first", "first"))
    second = self._get_column(kwargs.pop("second", "second"))
    result = diff_cols(first=first, second=second,offset_first=None, offset_second=None,**kwargs)
    return self._post_process(result, **kwargs)