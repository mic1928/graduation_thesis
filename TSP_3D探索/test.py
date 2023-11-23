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

def insert_at_position(lst, remove_length, remove_position, insert_position):
    # 準備
    lst_copy = lst.copy()
    minus = False
    if remove_length < 0:
        minus = True
        remove_length = -remove_length
    turn = True if (remove_position + remove_length > len(lst)) else False
    # リストから取り除く要素を取得
    double_lst = lst_copy + lst_copy
    removed_elements = double_lst[remove_position : (remove_position + remove_length)]
    # リストから要素を取り除く
    if turn:
        lst_copy = lst_copy[(remove_position + remove_length - len(lst)):remove_position]
    else:
        lst_copy = lst_copy[:remove_position] + lst_copy[(remove_position + remove_length):]
    # リストに要素を挿入する
    if minus:
        lst_copy = lst_copy[:insert_position] + removed_elements[::-1] + lst_copy[insert_position:]
    else:
        lst_copy = lst_copy[:insert_position] + removed_elements + lst_copy[insert_position:]
    return lst_copy

# テスト例
result1 = insert_at_position([10, 20, 30, 40, 50], 2, 3, 2)
print(result1)  # 出力: [10, 20, 50, 30, 40]

result2 = insert_at_position([10, 20, 30, 40, 50], 4, 2, -3)
print(result2)  # 出力: [20, 30, 40, 50, 10]

"""
# horizontal_under = 1.3
# horizontal_under = 0 if horizontal_under < 0 else horizontal_under
# horizontal_upper = 1 if horizontal_upper > 1 else horizontal_upper
# print(horizontal_under)

"""
import numpy as np
my_array = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
array2 = np.delete(my_array, -1, 1)
print(array2)
"""

# from cartesian_3d import Swap
# swsw = Swap([],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
# sec = swsw.assign_sector(0.8,0.5,0.7)
# print(sec)


from quasimc.sobol import Sobol
sobol = Sobol(3)
sobol_points = sobol.generate(100)
print(sobol_points.T[10])



