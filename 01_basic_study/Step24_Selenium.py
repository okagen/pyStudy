# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 10:24:39 2017

@author: 10007434
"""
# seleniumモジュールをインストール
## pip install -U selenium
# Chrome制御用のドライバをダウンロードしてexeを実行。9515ポートで待機してる状態になる。
## https://sites.google.com/a/chromium.org/chromedriver/downloads

from selenium import webdriver

from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
# ブラウザが立ち上がる
drv = webdriver.Remote("http://127.0.0.1:9515", DesiredCapabilities.CHROME)

# Amazonのトップページを表示
drv.get("http://www.amazon.co.jp")

# キーワードで商品検索
drv.find_element_by_xpath("//*[@id=\"twotabsearchtextbox\"]").send_keys("Python")
drv.find_element_by_xpath("//*[@id=\"nav-search\"]/form/div[2]/div/input").click()

# divタグの内容を取得
print(drv.find_element_by_css_selector("div").text)
