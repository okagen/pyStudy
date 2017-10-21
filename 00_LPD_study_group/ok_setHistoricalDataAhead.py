# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 13:40:59 2017

@author: 10007434
"""

from __future__ import unicode_literals
import copy
import datetime

def setHistoricalDataAhead(data, daysAhead=7):
    # ucdのデータをdeapcopyする。
    ahead = copy.deepcopy(data)
    #print(ucd_ahead.head())

    # dataはリストではなくDataFrame
    # マルチインデックスmultiindexをカラムcolumnに置きなおす。
    ahead.reset_index(inplace=True)
    #print(ahead.head())

    # aheadのDate部分に7日分を足す（7日分日付を進める）
    ahead["Date"] = ahead["Date"] + datetime.timedelta(days=daysAhead)
    #print(ahead.head())

    # indexを設定する
    ahead.set_index('Date', inplace=True)
    #print(ahead.head())

    return ahead
