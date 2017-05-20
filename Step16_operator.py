# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:43:23 2017

@author: 10007434
"""

#オペレータ試し
x = 350
y = 100

ope_pls = x + y
ope_minus = x -y
ope_multiplication = x * y
ope_division = x / y
ope_surplus = x % y
ope_trancated_division = x // y
ope_exponentiation = x ** y
ope_divmod = divmod(x,y)

#文字列の掛け算
z = '100' * 3

#文字列の演算
txt = "A,B,C,D,E,F,G"
txt_slice = txt[1:-1]
txt_len = len(txt)
txt_split = txt.split(",")
txt_join = "_".join(txt_split)
txt_lower = txt.lower()
txt_upper = txt.upper()
txt_find = txt.find("C")


