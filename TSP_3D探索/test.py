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

"""
from quasimc.sobol import Sobol
sobol = Sobol(3)
sobol_points = sobol.generate(100)
print(sobol_points.T[10])
"""
# import time
# from tqdm import tqdm
# from math import log10
# sum = 1
# for i in tqdm(range(1,100000)):
#     sum *= i
# print(log10(sum))

"""
lst:[0, 50, 36, 10, 57, 30, 16, 12, 21, 28, 56, 13, 62, 4, 15, 17, 38, 35, 61, 6, 9, 45, 
     41, 11, 23, 37, 20, 26, 47, 58, 59, 46, 3, 22, 31, 40, 24, 18, 48, 5, 
     29, 1, 63, 19, 51, 34, 49, 43, 55, 
     54, 33, 42, 39, 7, 27, 52, 2, 14, 32, 53, 44, 25, 60, 8]
"""

# def remove_extremes(lst):
#     if len(lst) <= 20:
#         return []

#     sorted_lst = sorted(lst)
#     trimmed_lst = sorted_lst[10:-10]

#     return trimmed_lst

# file_path = '/Users/tomo.f/Desktop/卒論/TSP_3D探索/mean.txt'

# # ファイルから数字を読み込んでリストに格納
# with open(file_path, 'r') as file:
#     numbers = [float(line.strip()) for line in file]

# numbers = remove_extremes(numbers)

# # 平均値を計算
# if numbers:
#     mean_value = (sum(numbers) / len(numbers))*100
#     print(f"平均値: {mean_value}")
# else:
#     print("ファイルに数字が含まれていません。")

"""
import numpy as np
import time
for i in range(100):
    time1 = time.time()
    r = np.array([0.0,0.0,0.0])
    for i in range(100):
        random_array = np.random.rand(100, 3)
        for i in range(100):
            r += random_array[i]
    print(r)

    time2 = time.time()
    r = np.array([0.0,0.0,0.0])
    for i in range(100):
        for i in range(100):
            random_array = np.random.rand(3)
            r += random_array
    print(r)

    time3 = time.time()
    time1_sum = time2 - time1
    time2_sum = time3 - time2
print(f"1: {time1_sum}, 2: {time2_sum}")
"""

# class Point:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def __repr__(self):
#         return f"Point(x={self.x}, y={self.y})"

# # インスタンスの作成
# p = Point(1, 2)

# # __repr__メソッドの呼び出し
# # print(repr(p))  # 出力: Point(x=1, y=2)
# print(p)

from openai import OpenAI

client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt="a white siamese cat",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url







