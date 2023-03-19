
# -*- coding: utf-8 -*-
from pandas import concat, DataFrame
from pandas_ta import Imports
from pandas_ta.overlap import ema
from pandas_ta.utils import get_offset, verify_series, signals
import numpy as np
import pandas as pd

def rollvals(close, length_list=None, offset=None, **kwargs):

    """Indicator: Moving Average, Convergence/Divergence (MACD)"""
    def ranking(s):
        return s.rank(ascending=False)[len(s)-1]
    
    # Validate arguments
    length_list = ['10min', '30min'] if length_list is None or len(length_list) ==0 else length_list
    length_str = 'None' if length_list is None or len(length_list) ==0 else ''.join(length_list)
    close = verify_series(close, )
    offset = get_offset(offset)

    if close is None: return
    
    _name = "ROLLVALS"
    _props = f"{length_str}_{offset}"
    eval_stmt = ''
    for lt, oper, agg in zip(length_list,kwargs['oper'], kwargs['aggs']):
        tmpst = f"close.rolling('{lt}', min_periods=1).{agg}() {oper}"
        eval_stmt = eval_stmt + tmpst

    print(eval_stmt[:-1])
    close = eval(eval_stmt[:-1])
 
    if offset != 0:
        close = close.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        close.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        close.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    close.name = f"{_name}_{_props}"
    close.category = "performance"
    
    return close


rollvals.__doc__ = \
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

def rollvals_method(self, length_list=None, offset=None, **kwargs):
    close = self._get_column(kwargs.pop("close", "close"))
    result = rollvals(close=close, length_list=length_list, offset=offset, **kwargs)
    return self._post_process(result, **kwargs)