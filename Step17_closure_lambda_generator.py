# -*- coding: utf-8 -*-
"""
Created on Sun May 28 09:37:53 2017

@author: 10007434
"""
# 関数内関数
def myName_innerFunc(given, family):
    gHead = 'My given name * 2 is = '
    fHead = 'My family name * 3 is = '
    def givenName():
        return (gHead + (given * 2))
    def familyName():
        return (fHead + (family * 3))
    
    iam = givenName() + " / " +familyName()
    print (iam)

print ('--- inner function ---')
myName_innerFunc('Hidehiro','Okamoto')
       
# クロージャー：関数を生成できる関数
# クロージャーとする場合は、関数内関数をreturnで返す。
def myName_closure(given, family):
    gHead = 'My given name * 2 is = '
    fHead = 'My family name * 3 is = '
    def givenName():
        return (gHead + (given * 2))
    def familyName():
        return (fHead + (family * 3))
    def prt():
        print (iam)
    
    iam = givenName() + " / " +familyName()
    return prt

print ('--- closure ---')
ok1 = myName_closure('Hidehiro', 'Okamoto')
ok1()

# 無形関数 lambda
def myName_lambda(given, gLambda, fLambda):
    gHead = 'My given name * 2 is = '
    fHead = 'My family name * 3 is = '
    def givenName():
        return (gHead + gLambda(given))
    def familyName():
        return (fHead + fLambda)
    def prt():
        print (iam)
    
    iam = givenName() + " / " +familyName()
    return prt

print ('--- lambda ---')
gLambda = (lambda g: (g+g))
fLambda = (lambda f: (f+f+f))("Okamoto")
ok2 = myName_lambda("Hidehiro", gLambda, fLambda)
ok2()

# ジェネレータ関数
# returnの代わりにyieldを使うと関数内でreturnが順番に生成されるイメージ。
# 呼出側ではforを使って順番に処理を呼び出すことが出来る。
def myName_generator(maxG, givenName, maxF, familyName):
    countG, countF = 1,1
    g,f = "",""
    while countG <= maxG:
        g = g + givenName
        yield g
        countG += 1
    while countF <= maxF:
        f = f + familyName
        yield f
        countF += 1

print ('--- generator ---')
for v in myName_generator(2, "Hidehiro", 3, "Okamoto"):
    print (v)

    