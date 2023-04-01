
# -*- coding: utf-8 -*-
from pandas import concat, DataFrame
from pandas_ta import Imports
from pandas_ta.overlap import ema
from pandas_ta.utils import get_offset, verify_series, signals
import numpy as np
import pandas as pd

def price_breach(close, length=None, offset=None, **kwargs):

    """Indicator: Moving Average, Convergence/Divergence (MACD)"""
    def ranking(s):
        return s.rank(ascending=False)[len(s)-1]
    
    # Validate arguments
    length = int(length) if length and length > 0 else 20
    
    close = verify_series(close, )
    offset = get_offset(offset)

    if close is None: return
    
    _name = "ROLLBRC"
    _props = f"{length}_{offset}"
    
    if kwargs["breach_type"] == 'morethan':
        close = close.rolling(length).apply(lambda x: (x[-1] > x[:-1]).sum()).fillna(0)
    elif kwargs["breach_type"] == 'lessthan':
        close = close.rolling(length).apply(lambda x: (x[-1] < x[:-1]).sum()).fillna(0)
    elif kwargs["breach_type"] == 'mean':
        close = close.rolling(length).apply(lambda x: (x > x[:].mean()).sum()).fillna(0).astype(int)
    elif kwargs["breach_type"] == 'min':
        close = close.rolling(length).apply(lambda x: (x > x[:].min()).sum()).fillna(0).astype(int)
    elif kwargs["breach_type"] == 'max':
        close = close.rolling(length).apply(lambda x: (x > x[:].max()).sum()).fillna(0).astype(int)
    elif kwargs["breach_type"] == 'median':
        close = close.rolling(length).apply(lambda x: (x > x[:].median()).sum()).fillna(0).astype(int)
    elif kwargs["breach_type"] == '10thquantile':
        close = close.rolling(length).apply(lambda x: (x > x[:].quantile(0.1)).sum()).fillna(0).astype(int)
    elif kwargs["breach_type"] == '25thquantile':
        close = close.rolling(length).apply(lambda x: (x > x[:].quantile(0.25)).sum()).fillna(0).astype(int)
    elif kwargs["breach_type"] == '75thquantile':
        close = close.rolling(length).apply(lambda x: (x > x[:].quantile(0.75)).sum()).fillna(0).astype(int)
    elif kwargs["breach_type"]  == '95thquantile':
        close = close.rolling(length).apply(lambda x: (x > x[:].quantile(0.95)).sum()).fillna(0).astype(int)
    else:
        close = (close.rolling(length).apply(lambda x: (x[-1] > x[:-1]).mean()).astype(int))    
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


price_breach.__doc__ = \
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

def price_breach_method(self, length=None, offset=None, **kwargs):
    close = self._get_column(kwargs.pop("close", "close"))
    result = price_breach(close=close, length=length, offset=offset, **kwargs)
    return self._post_process(result, **kwargs)