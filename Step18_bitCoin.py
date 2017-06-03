# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 08:47:31 2017

@author: 10007434
"""

# pip install requestsでインストールしておく
import requests
from pprint import pprint #整形表示メソッド

# https://lightning.bitflyer.jp/docs?lang=japanese
# http://techbureau-api-document.readthedocs.io/ja/latest/index.html
# https://coincheck.com/ja/documents/exchange/api

# ----------------------------------------------------
# Tickerを取得。
# requestsのgetメソッドを使ってbitFlyerのAPIを利用。Tickerを取得。
# product_code=通貨ペア
r = requests.get('https://api.bitflyer.jp/v1/ticker?product_code=BTC_JPY')
json_bitFlyer_ticker = r.json()
# timestamp=情報を生成したサーバタイム
# best_ask = 直近のASK価格
# best_bid = 直近のBID価格
# ltp = 最終約定価格
print('--- from json_bitFlyer_ticker ---')
pprint(json_bitFlyer_ticker)


# ZaifのAPIを利用。Tickerを取得。
r = requests.get('https://api.zaif.jp/api/1/ticker/btc_jpy')
json_zaif_ticker = r.json()
print('--- from json_zaif_ticker ---')
pprint(json_zaif_ticker)

# coincheckのAPIを利用。Tickerを取得。
r = requests.get('https://coincheck.com/api/ticker')
json_coincheck_ticker = r.json()
print('--- from json_coincheck_ticker ---')
pprint(json_coincheck_ticker)

# ----------------------------------------------------
# 板情報を取得。
r = requests.get('https://api.bitflyer.jp/v1/getboard?product_code=BTC_JPY')
json_bitFlyer_board = r.json()

r = requests.get('https://api.zaif.jp/api/1/depth/btc_jpy')
json_zaif_board = r.json()

r = requests.get('https://coincheck.com/api/order_books')
json_coincheck_board = r.json()
