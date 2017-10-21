# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです
"""

# ===============================
# リスト 
# 古典的なリスト作成
classical_list = []
for n in range(0,5):
    classical_list.append(n+1)

# python風リスト作成 リスト内包表記
python_list = [n+1 for n in range(0,5)]

# リストの中にリストを設定
listlist = [[2,1],[3,3],[1,0],[20,5],[100,6]]

# リストの中のリストの、前の要素を後ろの要素で割る。
# 後ろの要素が０より大きい場合のみ。
listDivList = [x/y for x,y in listlist if y>0]

# ================================
# 辞書
# {[キー：値],[キー：値],・・・}
# キーが重複した場合、後からの要素で上書きされる。
dic1 = {"apple":"RINGO","oange":"MIKAN","banana":"BANANA","potate":"IMO"}
# dic1の中の要素を取り出し e,j とする。
# eの中にoが含まれていたら、eを大文字化、jを小文字化してリストを作り直す。
dic2 = {e.upper():j.lower() for e,j in dic1.items() if 'o' in e}

# ================================
# タプル
# (要素,要素,要素,・・・)
# 後から要素を変えられない。
tup = (1,2,3,4,5)
#tup[1] = 3 # エラーになる

# ================================
# 集合
# 辞書のキーだけのようなもの。要素が重複した場合、１つを残して削除される。
mass = {"okagen","okamoto","hide","hiro","hidehiro", "okamoto"}
print("\n--- mass ---")
print(mass)

# ================================
# 集合の演算
# 数字部分は2回のテストの成績。どちらかのテストが70点以下、かつ平均が70以上の要素を抜き出す
scores = {("okagen",70,50),("okamoto",20,80),("hide",50,90),("hiro", 100, 90),("hidehiro", 55,45)}
print("\n--- scores ---")
print(scores)         

#2回目が70点以上
f70 = {(name, firstScore, secondScore) 
        for name, firstScore, secondScore in scores if firstScore>=70}
print("\n--- forstScore >= 70 ---")
print(f70)

#2回目が70点以上
s70 = {(name, firstScore, secondScore) 
        for name, firstScore, secondScore in scores if secondScore>=70}
print("\n--- secondScoe >= 70 ---")
print(s70)

#平均が70以上
av70 = {(name, firstScore, secondScore) 
        for name, firstScore, secondScore in scores if (firstScore + secondScore)>=140}
print("\n--- average >= 70 ---")
print(av70)

# どちらかのテストが70点以下、かつ平均が70以上の要素
# ^ は共通部分を除いた部分
# & は共通部分
res = (f70 ^ s70) & av70
print("\n--- rsult ---")
print(res)






