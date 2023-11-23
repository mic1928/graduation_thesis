# def find_first_index(lst, target):
#     try:
#         index = lst.index(target)
#         return index
#     except ValueError:
#         return None

# # 例としてリストと検索対象の要素を指定
# my_list = [10, 20, 30, 40, 20, 50]
# target_element = 20

# # 関数を呼び出して結果を取得
# result_index = find_first_index(my_list, target_element)

# # 結果を表示
# if result_index is not None:
#     print(f"最初に一致する要素のインデックス: {result_index}")
# else:
#     print("要素が見つかりませんでした。")

"""
def insert_at_position(lst, remove_position, insert_position, remove_length):
    # リストから取り除く要素を取得
    removed_elements = lst[remove_position : (remove_position + remove_length) % len(lst)]

    # リストから要素を取り除く
    lst = lst[:remove_position] + lst[(remove_position + remove_length) % len(lst):]

    # リストに要素を挿入する
    lst = lst[:insert_position] + removed_elements + lst[insert_position:]

    return lst

# テスト例
result1 = insert_at_position([10, 20, 30, 40, 50], 2, 3, 2)
print(result1)  # 出力: [10, 20, 50, 30, 40]

result2 = insert_at_position([10, 20, 30, 40, 50], 4, 3, 2)
print(result2)  # 出力: [20, 30, 40, 50, 10]
"""
# horizontal_under = 1.3
# horizontal_under = 0 if horizontal_under < 0 else horizontal_under
# horizontal_upper = 1 if horizontal_upper > 1 else horizontal_upper
# print(horizontal_under)

import numpy as np
my_array = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
array2 = np.delete(my_array, -1, 1)
print(array2)





