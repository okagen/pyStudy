def module_add(x, y):
    return x + y

def module_sub(x, y):
    return x - y

# __name__はグローバル変数。自身がスクリプトファイルとして実行されている場合は
# __name__ == "__main__" となる。
# モジュールとして実行されている場合は
# __name__ == "Step12_module" となる。
print("\n This message is from Step12_module.py [ptrint(__name__)] : " + __name__)

#この条件を設定しておくと、モジュールとして実行された際に以下が呼ばれない。
if __name__ == "__main__":
    ans = module_add(3, 4)
    print(ans)
    ans = module_sub(3, 4)
    print(ans)
    
