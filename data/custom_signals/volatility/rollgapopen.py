
# -*- coding: utf-8 -*-
from pandas import concat, DataFrame
from pandas_ta import Imports
from pandas_ta.overlap import ema
from pandas_ta.utils import get_offset, verify_series, signals
import numpy as np
import pandas as pd

def rollgapopen(close, frequency=None, offset=None, **kwargs):

    """Indicator: Moving Average, Convergence/Divergence (MACD)"""
 
    # Validate arguments
    close = verify_series(close, )
    offset = get_offset(offset)

    if close is None: return
    _name = "ROLLGO"
    _props = f"_{frequency}"
    frequency = 'd' if frequency is None else frequency
    merge_dict={}
    close = close.resample(frequency).bfill().groupby(pd.Grouper(freq=frequency)).apply(lambda x: x[0]).subtract(
            close.resample(frequency).ffill().groupby(pd.Grouper(freq=frequency)).apply(lambda x: x[-1]).fillna(0))
    merge_dict.update({f"{_name}_{_props}": close[1:]})
    close = pd.concat(merge_dict, axis=1)
    if offset != 0:
        close = close.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        close.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        close.fillna(method=kwargs["fill_method"], inplace=True)
        
    # Prepare DataFrame to return
    close.name = f"{_name}_{_props}"
    close.category = "volatility"
    return close

rollgapopen.__doc__ = \
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

def rollgapopen_method(self, frequency=None, offset=None, **kwargs):
    close = self._get_column(kwargs.pop("close", "close"))
    result = rollgapopen(close=close, frequency=frequency, offset=offset, **kwargs)
    return self._post_process(result, **kwargs)