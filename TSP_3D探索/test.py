def find_first_index(lst, target):
    try:
        index = lst.index(target)
        return index
    except ValueError:
        return None

# 例としてリストと検索対象の要素を指定
my_list = [10, 20, 30, 40, 20, 50]
target_element = 20

# 関数を呼び出して結果を取得
result_index = find_first_index(my_list, target_element)

# 結果を表示
if result_index is not None:
    print(f"最初に一致する要素のインデックス: {result_index}")
else:
    print("要素が見つかりませんでした。")

