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

# 条件分岐
l = ["aaa","bbb","ccc","ddd","eee"]

# if文
print("---[if]---")
if l[0] == "bbb":
    print(l[1])
elif l[2] == "ccc":
    print(l[2])
else:
    print(l)

# while文
print("---[while]---")
i=0
while i<3:
    print(l[i])
    i=i+1
    
# for文
print("---[for]---")
for n in l:
    print(n)
    
# for文 + break
print("---[for + break]---")
for n in l:
    if n == "ddd":
        break
    print(n)
    
# for文 + continue
print("---[for + continue]---")
for n in l:
    if n == "ddd":
        continue
    print(n)

# 位置引数
print("---[psition argument]---")
def position(a,b):
    print(a+"+"+b)
    
position("okamoto", "hidehiro")
position(b="okamoto", a="hidehiro")

def position2(a,b,c="ucd"):
    print(a+"+"+b+"+"+c)

position2("okamoto", "work for")
position2("okamoto", "had worked for", "KKE")

# タプル引数
# 引数に*を付けると、可変長引数を受け取ることが出来る。
print("---[function with tuple argument]---")
def tupleFunc1(t):
    print(t)

def tupleFunc2(*t):
    print(t)
    
tupleFunc1(l)
tupleFunc2(l)
tupleFunc2("a","b",l)

# 辞書引数
# 引数に**を付けると、辞書になる。
print("---[function with dictionary argument]---")
def dictFunc(**d):
    print(d)

dictFunc(a="aaa",b="bbb", c=l)

# グロ^バル変数を変更する
print("---[Original gloval variable]---")
glb = "This is a gloval variable"
print(glb)
def chgGlb(a):
    global glb # 変数を変えたいときは、globalを宣言する。
    glb = glb + a
    print(glb)
print("---[change gloval variable]---")
chgGlb(", you know.")
print("---[changeed gloval variable]---")
print(glb)




    