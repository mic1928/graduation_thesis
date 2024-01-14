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

"""
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
"""

# def read_atsp_file(file_path):
#     matrix = []
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#     lines = lines[:-1]

#     # Find the start of the data section
#     data_start_index = lines.index("EDGE_WEIGHT_SECTION\n") + 1

#     # Read data and convert to a 2D list
#     for i in range(data_start_index, len(lines)):
#         row = list(map(int, lines[i].split()))
#         matrix.append(row)

#     return matrix

# # Replace 'path/to/br17.atsp' with the actual path to your file
# file_path = '/Users/tomo.f/Desktop/br17.atsp'
# result_matrix = read_atsp_file(file_path)

# # Print the resulting 2D list
# for row in result_matrix:
#     print(row)

"""
def read_atsp_file(file_path):
    matrix = []
    dimension = 0
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the dimension
    for line in lines:
        if line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1].strip())
            break

    # Find the start of the data section
    data_start_index = lines.index("EDGE_WEIGHT_SECTION\n") + 1

    # Read data and convert to a 2D list
    for i in range(data_start_index, data_start_index + dimension):
        row = list(map(int, lines[i].split()))
        matrix.append(row)

    return matrix

# Replace 'path/to/br17.atsp' with the actual path to your file
file_path = '/Users/tomo.f/Desktop/br17.atsp'
result_matrix = read_atsp_file(file_path)

# Print the resulting 2D list
for row in result_matrix:
    print(row)
"""



# # Replace 'path/to/br17.atsp' with the actual path to your file
# file_path = '/Users/tomo.f/Desktop/br17.atsp'
# result_numbers = read_atsp_file(file_path)

# # Print the resulting list
# print(result_numbers)
# print(len(result_numbers[0]))

"""
from week6_3d import calculate_total_distance, read_atsp_file


def two_opt(tour, cost_matrix):
    num_cities = len(tour)
    improvement = True

    while improvement:
        improvement = False

        for i in range(1, num_cities - 1):
            for j in range(i + 1, num_cities):
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                # print(new_tour)
                new_length = calculate_total_distance(cost_matrix, new_tour)

                if new_length < calculate_total_distance(cost_matrix, tour):
                    tour = new_tour
                    improvement = True

    return tour

file_path = 'ATSPlib/ft70.atsp'
# 例: 非対称な移動コスト
cost_matrix = read_atsp_file(file_path)
# print("cost_matrix",len(cost_matrix))

# 例: 初期の巡回路
initial_tour = list(range(len(cost_matrix)))

# 2-opt法による改善
optimized_tour = two_opt(initial_tour, cost_matrix)

# 結果の表示
print("Initial Tour:", initial_tour)
print("Optimized Tour:", optimized_tour)
print("Initial Tour Length:", calculate_total_distance(cost_matrix, initial_tour))
print("Optimized Tour Length:", calculate_total_distance(cost_matrix, optimized_tour))
"""

# def solve_tsp_simulated_annealing(
#     distance_matrix: np.ndarray,
#     x0: Optional[List[int]] = None,
#     perturbation_scheme: str = "two_opt",
#     alpha: float = 0.9,
#     max_processing_time: Optional[float] = None,
#     log_file: Optional[str] = None,
#     verbose: bool = False,
# ) -> Tuple[List, float]:


#     x, fx = setup_initial_solution(distance_matrix, x0)
#     temp = _initial_temperature(distance_matrix, x, fx, perturbation_scheme)
#     max_processing_time = max_processing_time or inf
#     log_file_handler = (
#         open(log_file, "w", encoding="utf-8") if log_file else None
#     )

#     n = len(x)
#     k_inner_min = n
#     k_inner_max = MAX_INNER_ITERATIONS_MULTIPLIER * n
#     k_noimprovements = 0  # number of inner loops without improvement

#     tic = default_timer()
#     stop_early = False
#     while (k_noimprovements < MAX_NON_IMPROVEMENTS) and (not stop_early):
#         k_accepted = 0  # number of accepted perturbations
#         for k in range(k_inner_max):
#             if default_timer() - tic > max_processing_time:
#                 _print_message(TIME_LIMIT_MSG, verbose, log_file_handler)
#                 stop_early = True
#                 break

#             xn = _perturbation(x, perturbation_scheme)
#             fn = compute_permutation_distance(distance_matrix, xn)

#             if _acceptance_rule(fx, fn, temp):
#                 x, fx = xn, fn
#                 k_accepted += 1
#                 k_noimprovements = 0

#             msg = (
#                 f"Temperature {temp}. Current value: {fx} "
#                 f"k: {k + 1}/{k_inner_max} "
#                 f"k_accepted: {k_accepted}/{k_inner_min} "
#                 f"k_noimprovements: {k_noimprovements}"
#             )
#             _print_message(msg, verbose, log_file_handler)

#             if k_accepted >= k_inner_min:
#                 break

#         temp *= alpha  # temperature update
#         k_noimprovements += k_accepted == 0

#     if log_file_handler:
#         log_file_handler.close()

#     return x, fx


"""
import matplotlib.pyplot as plt

# 入力例
cities = [101, 264, 318, 654, 1655]
# cities = [39, 70, 323, 358, 443]
execution_times = [[0.004 , 0.03 , 0.04 , 0.2 , 1.0], [0.008 , 0.05 , 0.07 , 0.3 , 2.0],[4.2 , 47.9 , 75.3 , 603.4 , 5264.4],[0.02 , 0.2 , 0.9 , 8.1 , 150.4],[0.2 , 2.9 , 4.5 , 40.3 , 3853.3],[984.8 , 1918.2 , 2435.3 , 7926.9 , 29032.1]]
# execution_times = [[0.001 , 0.002 , 0.03 , 0.05 , 0.07], [0.002 , 0.005 , 0.05 , 0.07 , 0.1],[0.5 , 1.8 , 47.7 , 62.7 , 113.7],[0.03 , 0.1 , 17.8 , 25.4 , 50.7],[386.3 , 804.6 , 1050.6 , 1259.4 , 1467.7]]
labels = ["Nearest Neighbor","2-app","Simulated Annealing","2-opt","Lin-Kernighan","Relaxation"]
# labels = ["Nearest Neighbor","2-app","Simulated Annealing","2-opt","Relaxation"]

# グラフの描画
plt.figure(figsize=(8, 6))

# 各系列のプロット
for i, series in enumerate(execution_times):
    # plt.scatter(cities, series, label=labels[i], marker='o')  # マーカーを変更する場合は'o'を変更
    plt.plot(cities, series, label=labels[i], marker='o', linestyle='-')

# 軸ラベルとタイトルの設定
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Cities')
plt.ylabel('Execution Time (seconds)')
# plt.title('Relationship between the number of cities and execution time')

# 凡例の表示
plt.legend()

# グリッドの表示
plt.grid(True)

# グラフの表示
plt.show()
"""

# # file_path = "/Users/tomo.f/Desktop/卒論/完成版/実行結果/提案手法/eil101_tsp.txt"
# # file_path = "/Users/tomo.f/Desktop/卒論/完成版/実行結果/提案手法/pr264_tsp.txt"
# # file_path = "/Users/tomo.f/Desktop/卒論/完成版/実行結果/提案手法/lin318_tsp.txt"
# file_path = "/Users/tomo.f/Desktop/卒論/完成版/実行結果/提案手法/p654_tsp.txt"
# # file_path = "/Users/tomo.f/Desktop/卒論/完成版/実行結果/提案手法/d1655_tsp.txt"
# # file_path = "/Users/tomo.f/Desktop/卒論/完成版/実行結果/提案手法/rbg443_atsp.txt"

# try:
#     with open(file_path, 'r') as file:
#         # ファイルから全ての行を読み込んでリストに格納
#         content = file.readlines()

#     length_list = []
#     time_list = []
#     for line in content:
#         if line[0] != "i":
#             continue
#         else:
#             length_list.append(float(line.split('length:')[1].split(' ')[0]))
#             time_list.append(float(line.split('lap_time:')[1]))

#     length_list = length_list[-200:]
#     time_list = time_list[-200:]
            


#     print(length_list)
#     print(len(length_list))
#     print(time_list)
#     print(len(time_list))

# except FileNotFoundError:
#     print(f"指定されたファイル '{file_path}' が見つかりませんでした。")
# except Exception as e:
#     print(f"エラーが発生しました: {e}")

# import matplotlib.pyplot as plt

# def plot_line_chart(x_values, y_values):
#     """
#     x_values: x軸の値を格納したリスト
#     y_values: y軸の値を格納したリスト
#     x_label: x軸のラベル
#     y_label: y軸のラベル
#     title: グラフのタイトル
#     """
#     plt.plot(x_values, y_values, linestyle='-', label='Relaxation')
#     # plt.axhline(y=629, color='r', linestyle='--', label='Exact Solution')
#     # plt.axhline(y=49135, color='r', linestyle='--', label='Exact Solution')
#     # plt.axhline(y=42029, color='r', linestyle='--', label='Exact Solution')
#     plt.axhline(y=34643, color='r', linestyle='--', label='Exact Solution')
#     # plt.axhline(y=62128, color='r', linestyle='--', label='Exact Solution')
#     # plt.axhline(y=2720, color='r', linestyle='--', label='Exact Solution')
#     plt.xlabel("Execution Time (seconds)")
#     plt.ylabel("Tour Length")
#     # plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# plot_line_chart(time_list, length_list)


import numpy as np
import matplotlib.pyplot as plt

# 入力リスト
# x_data = [101, 264, 318, 654, 1655]
# y_datas = [[0.004 , 0.03 , 0.04 , 0.2 , 1.0], [0.008 , 0.05 , 0.07 , 0.3 , 2.0],[4.2 , 47.9 , 75.3 , 603.4 , 5264.4],[0.02 , 0.2 , 0.9 , 8.1 , 150.4],[0.2 , 2.9 , 4.5 , 40.3 , 3853.3],[984.8 , 1918.2 , 2435.3 , 7926.9 , 29032.1]]
# labels = ["Nearest Neighbor      ","2-app                        ","Simulated Annealing","2-opt                         ","Lin-Kernighan            ","Relaxation                 "]

x_data = [39, 70, 323, 358, 443]
y_datas = [[0.001 , 0.002 , 0.03 , 0.05 , 0.07], [0.002 , 0.005 , 0.05 , 0.07 , 0.1],[0.5 , 1.8 , 47.7 , 62.7 , 113.7],[0.03 , 0.1 , 17.8 , 25.4 , 50.7],[386.3 , 804.6 , 1050.6 , 1259.4 , 1467.7]]
labels = ["Nearest Neighbor      ","2-app                        ","Simulated Annealing","2-opt                         ","Relaxation                 "]

# データをnumpy配列に変換し、両対数変換を行う
log_x = np.log10(x_data)
log_ys = np.array([np.log10(y_data) for y_data in y_datas])

# 線形回帰を行う
coeffss = np.array([np.polyfit(log_x, log_y, 1) for log_y in log_ys])

# 近似直線の式を作成する
equations = [f"y = {10 ** coeffs[1]:.10f}x^{coeffs[0]:.6f}" for coeffs in coeffss]


x_data_extention = [x_data[0], x_data[-1]*6]
# 元のデータと近似直線をプロットする
plt.figure(figsize=(8, 8))
for i, y_data in enumerate(y_datas):
    plt.scatter(x_data, y_data, marker='o')
    plt.plot(x_data_extention, 10 ** coeffss[i][1] * np.array(x_data_extention) ** coeffss[i][0], label=f'{labels[i]:<20} Fitted line:{equations[i]}')

# plt.plot(x_data, 10 ** coeffs[1] * np.array(x_data) ** coeffs[0], label=f'Fitted line:{equation}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Cities')
plt.ylabel('Execution Time (seconds)')
# plt.title('2D log-log plot with fitted line')
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.legend(loc='upper center', bbox_to_anchor=(.5, -.10))
plt.tight_layout()
plt.grid(True)
plt.show()

