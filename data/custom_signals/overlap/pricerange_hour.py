
# -*- coding: utf-8 -*-
from pandas import concat, DataFrame
from pandas_ta import Imports
from pandas_ta.overlap import ema
from pandas_ta.utils import get_offset, verify_series, signals
import numpy as np
import pandas as pd

def pricerange_hour(first, second,length=None, offset=None, **kwargs):

    """Indicator: Moving Average, Convergence/Divergence (MACD)"""
    def ranking(s):
        return s.rank(ascending=False)[len(s)-1]
    
    # Validate arguments
    length = int(length) if length and length > 0 else 20
    
    first = verify_series(first, )
    second = verify_series(second, )
    offset = get_offset(offset)

    if first is None or second is None: return
    
    range_type = kwargs['range_type']
    r1 = kwargs.get('hour_range')[0]
    r2 = kwargs.get('hour_range')[1]
    _name = "ROLLPDR"
    _props = f"_{length}_{offset}_{range_type}_{r1.replace(':','')}_{r2.replace(':','')}".replace("-", '_minus_')
    
    if range_type == 'price_range':
        close = first.between_time(r1, r2).groupby(pd.Grouper(freq='d')).max() - second.between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
    elif range_type == 'price_deviation_max_first_col':
        close = first.between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - second.between_time(r1, r2).groupby(pd.Grouper(freq='d')).max()
    elif range_type == 'price_deviation_min_first_col':
        close = first.between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - second.between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
    elif range_type == 'price_deviation_max_second_col':
        close = first.between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - second.between_time(r1, r2).groupby(pd.Grouper(freq='d')).max()
    elif range_type == 'price_deviation_min_second_col':
        close = first.between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - second.between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
    else:
        close = first.between_time(r1, r2).groupby(pd.Grouper(freq='d')).max() - second.between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
 
    if offset != 0:
        close = close.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        close.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        close.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    close.name = f"{_name}_{_props}"
    close.category = "overlap"
    
    return close


pricerange_hour.__doc__ = \
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

def pricerange_hour_method(self, length=None, offset=None, **kwargs):
    first = self._get_column(kwargs.pop("first", "first"))
    second = self._get_column(kwargs.pop("second", "second"))
    result = pricerange_hour(first=first, second=second,length=length, offset=offset, **kwargs)
    return self._post_process(result, **kwargs)