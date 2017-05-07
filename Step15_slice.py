# -*- coding: utf-8 -*-

# Rangeの第2要素は0から数えたindex番号。と考える。
contents_n19 = [n for n in range(-1,9)]
contents_09 = [n for n in range(0,9)]
contents_19 = [n for n in range(1,9)]

# スライス１：index=1(第2要素)から0スタートのindex=4(第5要素)まで
sl_0_5 = contents_19[0:5]

# スライス2：index=1(第2要素)から0スタートのindex=4(第5要素)まで
sl_1_5 = contents_19[1:5]

# スライス3：index=1(第2要素)から最終要素の一つ手前までを取得
leng = len(contents_19)
sl_1_lenn1 = contents_19[1:leng-1]

# スライス4：index=1(第2要素)から最終要素の一つ手前までを取得
sl_1_n1 = contents_19[1:-1]

# スライス5：全要素取得
sl_all = contents_19[:]

# スライスの代入 参照とコピー
print('contents_19')
print(contents_19)

contents_ref = contents_19
print('contents_ref = contents_19')
print(contents_ref)

contents_copy = contents_19[:]
print('contents_copy = contents_19[:]')
print(contents_copy)

contents_19 = contents_09[:]
print('contents_19 = contents_09[:]')
print(contents_19)

print('contents_ref 参照代入なので変わるはずだけど変わらない')
print(contents_ref)

print('contents_copy')
print(contents_copy)
