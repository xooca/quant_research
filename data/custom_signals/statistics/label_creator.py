
# -*- coding: utf-8 -*-
from pandas import concat, DataFrame
from pandas_ta import Imports
from pandas_ta.overlap import ema
from pandas_ta.utils import get_offset, verify_series, signals
import numpy as np
import pandas as pd

def label_generator_14class(val):
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

def label_generator_9class(val):
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

def label_generator_7class(val):
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

def label_generator_4class( val):
    if val <= 30 and val >= -30:
        return 'neutral'
    elif val > 30:
        return 'call'
    elif val < -30:
        return 'put'
    else:
        return 'unknown'


def label_generator_4class_50points( val):
    if val <= 50 and val >= -50:
        return 'neutral'
    elif val > 50:
        return 'call'
    elif val < -50:
        return 'put'
    else:
        return 'unknown'

def label_generator_3class_50points_no_neutral(val):
    if val > 50:
        return 'call'
    elif val < -50:
        return 'put'
    else:
        return 'unknown'

def label_generator_3class_20points_no_neutral(val):
    if val > 20:
        return 'call'
    elif val < -20:
        return 'put'
    else:
        return 'unknown'
                
def label_generator_4class_mod1(val):
    if val <= 20 and val >= -20:
        return 'neutral'
    elif val > 20:
        return 'call'
    elif val < -20:
        return 'put'
    else:
        return 'unknown'

def label_generator_4class_mod2(val):
    if val <= 20 and val >= -20:
        return 'neutral'
    elif val > 30:
        return 'call'
    elif val < -30:
        return 'put'
    else:
        return 'unknown'


def label_creator(close,length=None, offset=None, **kwargs):

    """Indicator: Moving Average, Convergence/Divergence (MACD)"""
    
    # Validate arguments
    length = int(length) if length and length > 0 else 20
    
    close = verify_series(close, )
    offset = get_offset(offset)

    if close is None: return
    
    _name = "LABEL"
    _props = f"_{length}_{kwargs['apply_func']}".replace("-", '_minus_')
    
    close = close.shift(length).subtract(close).apply(kwargs['apply_func']) 
    if offset != 0:
        close = close.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        close.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        close.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    close.name = f"{_name}_{_props}"
    close.category = "statistics"
    
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

def label_creator_method(self, length=None, offset=None, **kwargs):
    close = self._get_column(kwargs.pop("close", "close"))
    result = label_creator(close,length=length, offset=offset, **kwargs)
    return self._post_process(result, **kwargs)