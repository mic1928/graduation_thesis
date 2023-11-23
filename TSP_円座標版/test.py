# from circular_coordinates import swap_elements
def rotate_to_smallest(tour):  #直接は関係ない
    min_index = tour.index(min(tour))
    rotated_tour = tour[min_index:] + tour[:min_index]
    if rotated_tour[1] > rotated_tour[-1]:
        rotated_tour[1:] = rotated_tour[1:][::-1]
    return rotated_tour

def swap_elements(lst, i, j):
    lst_copy = lst.copy()
    # リストの要素を入れ替える
    lst_copy[i], lst_copy[j] = lst_copy[j], lst_copy[i]
    return lst_copy

def make_swapped_list(lst):  #直接は関係ない
    swapped_list_array = []
    for i in range(1,len(lst)):
        swapped_list = swap_elements(lst, 0, i)
        swapped_list_array.append(swapped_list)
    return swapped_list_array

def all_tour(N:int, seach_depth:int = None):  #直接は関係ない
    if seach_depth is None:
        seach_depth = N
    tour_array = []
    original_list = range(1,N+1)
    tour_array.append(tuple(original_list))
    for i in range(seach_depth):
        # print(f"{i}周目")
        temp_tour_array = []
        for tour in tour_array:
            swapped_list = make_swapped_list(list(tour))
            for swapped in swapped_list:
                roteted_swapped = rotate_to_smallest(swapped)
                temp_tour_array.append(tuple(roteted_swapped))
        tour_array = list(tour_array) + temp_tour_array
        tour_array = set(tour_array)
    return tour_array

# def coverage_ratio(N:int, depth:int=None):  #直接は関係ない
#     all = all_tour(N)
#     if depth is None:
#         for j in range(N):
#             halfsearch = all_tour(N,j)
#             print(f"都市数が{N:>2}で探索深さが{j:>2}の時、網羅率は{len(halfsearch)/len(all)}です")
#     else:
#         halfsearch = all_tour(N,depth)
#         print(f"都市数が{N:>2}で探索深さが{depth:>2}の時、網羅率は{len(halfsearch)/len(all)}です")

"""
class Circular:
    def __init__(self, num_cities:int, search_depth:int=None, *args, **kwargs):
        self.num_cities = num_cities
        if search_depth is None:
            self.search_depth = int(num_cities**(1/2))

    def 
"""
# import math
# def sum_of_equiproportional(equal_ratio: int, number_of_terms: int)->float: #Equal_ratio:公比a, number_of_term:項数n
#     a = equal_ratio
#     n = number_of_terms
#     # return math.log10((a**(n+1)-1)/(a-1)) #正確だけど数が大きすぎると計算できない
#     return (n+1) * math.log10(a) - math.log10(a-1)  #不正確だけど大きい数でも計算できる

# def boundary_list_for_equiproportion(ratio: int, max_searchdepth: int)->list: #ratio:都市数-1, max_searchdepth:最大探索深さ
#     boundary_list = [sum_of_equiproportional(ratio, depth) for depth in range(max_searchdepth+1)]
#     normalized_boundary_list = [element - boundary_list[-1] for element in boundary_list]
#     return normalized_boundary_list

# print(sum_of_equiproportional(5,5))
# print(boundary_list_for_equiproportion(1024,30))

"""
import random
import math

def generate_exponential_interval_random(num_cities, search_depth):
    time_0 = 0
    time_1 = 0
    time_2 = 0

    # 区間ごとの確率を計算
    probabilities = [num_cities**i / sum([num_cities**j for j in range(search_depth)]) for i in range(search_depth)]
    # print(f"probabilities:{probabilities}")

    for i in range(10000000):
        # 区間の選択
        selected_interval = random.choices(range(search_depth), probabilities)[0]
        if selected_interval == 1:
            time_1 += 1
        elif selected_interval == 0:
            time_0 += 1

    return [time_0, time_1, 10000000-time_0-time_1]

# 例: 区間の数を指定

# for i in range(100000):
#     random_number = generate_exponential_interval_random(1024, 3)
#     if random_number == "珍しい！":
#         print(i, "珍しい！")
#         break
#     print(i, random_number)
print(generate_exponential_interval_random(1024, 3))
"""

# def swap_elements(lst, index1, index2):
#     lst_copy = lst.copy()
#     if 0 <= index1 < len(lst_copy) and 0 <= index2 < len(lst_copy):
#         # 指定されたインデックスが範囲内にあるか確認
#         lst_copy[index1], lst_copy[index2] = lst_copy[index2], lst_copy[index1]
#         return lst_copy
#     else:
#         print("Invalid index values. Please provide valid indices.")
#         return None
"""
def cumulative_sum(input_list):
    # 累積和を格納するリスト
    result_list = [0]

    # 入力リストの要素を順番に加算して累積和を計算
    current_sum = 0
    for number in input_list:
        current_sum += number
        result_list.append(current_sum)

    return result_list
"""
# def to_base_n(number, n, depth):
#     if n < 2:
#         raise ValueError("nは2以上の整数である必要があります")

#     remainder = [number % n]
#     for i in range(depth):
#         number //= n
#         remainder.append(number % n)
#     # # while number > 0:
#     #     number //= n
#     #     remainder.append(number % n)
#     return remainder[:-1][::-1]

# # 例: 入力が 94 で n が 4 の場合
# number_input = 6
# n_value = 4
# result_base_n = to_base_n(number_input, n_value, 4)
# print(result_base_n)
"""
from circular_coordinates import random_number_to_depth_equal_probabilities
for i in range(100):
    print(i,random_number_to_depth_equal_probabilities(0.01*i, 3))
"""
# from skopt import gp_minimize
# import numpy as np
# def func(param=None):
#     ret = np.cos(param[0] + 2.34) + np.cos(param[1] - 0.78)
#     print(f"param[0]:{param[0]}, param[1]:{int(param[1])}")
#     return -ret 

# if __name__ == '__main__':
#     x1 = (-np.pi, np.pi)
#     x2 = (-np.pi, np.pi)
#     x = (x1, x2)

#     # x = (0,x2)
#     result = gp_minimize(func, x, 
#                           acq_func="EI",
#                           n_calls=30,
#                         #   n_initial_points = 10
#                           noise=0.0,
#                           model_queue_size=1,
#                           verbose=True)
#     print(result)
"""
from skopt import gp_minimize

def objective_function(x=None):
    # Your optimization logic here, for example, a simple quadratic function
    result = x[0]**2 + x[1]**2
    return result

# Define the search space
dimensions = ((-1.0, 1.0), (-1.0, 1.0))

# Run gp_minimize
result = gp_minimize(objective_function, dimensions, verbose=True)

# Print the results
print("Best parameters:", result.x)
print("Minimum value:", result.fun)
"""

# from skopt import gp_minimize
# import matplotlib.pyplot as plt
# from skopt.plots import plot_convergence

# # Define your objective function
# def objective_function(x):
#     result = x[0]**2 + x[1]**2
#     return result

# # Define the search space
# dimensions = [(-1.0, 1.0), (-1.0, 1.0)]

# # Callback function to plot optimization progress
# def callback_plot(*args):
#     res = args[0]  # The first argument is the optimization result
#     plt.scatter(len(res.func_vals), res.fun, color='red')  # Plot the current best point in red

# # Run gp_minimize with the callback
# result = gp_minimize(objective_function, dimensions=dimensions, callback=callback_plot, verbose=True)

# # Plot the convergence plot
# plot_convergence(result)
# plt.show()

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_heatmap(coordinates, values):
    # Extract x and y coordinates from the list of 2D coordinates
    x_coords, y_coords = zip(*coordinates)

    # Define a grid for the heatmap
    x_grid, y_grid = np.mgrid[min(x_coords):max(x_coords):500j, min(y_coords):max(y_coords):500j]

    # Interpolate the values to the grid
    grid_points = np.vstack((x_coords, y_coords)).T
    interpolated_values = griddata(grid_points, values, (x_grid, y_grid), method='cubic')

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(interpolated_values.T, extent=(min(x_coords), max(x_coords), min(y_coords), max(y_coords)),
               origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='Objective Function Value')
    plt.scatter(x_coords, y_coords, c=values, cmap='viridis', edgecolors='k', linewidths=0.5)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Heatmap of Objective Function')
    plt.show()

# Example usage:
coordinates = [[0.8342135867835497, 2.167512810163452], [0.6119102397235746, 4.6322291090137195], [0.0, 5.427096199304286],[0.7409996933597347, 0.01]]  # Your list of coordinates
values = np.array([1.915e+04, 1.838e+04, 9.758e+03, 1.193e+04])  # Your list of objective function values

plot_heatmap(coordinates, values)
"""

# import numpy as np

# def polar_to_cartesian(r, theta):
#     x = r * np.cos(theta)
#     y = r * np.sin(theta)
#     return x, y

# # Example usage:
# r = 1
# theta = np.pi/2

# x, y = polar_to_cartesian(r, theta)
# print(f"Output: [{x}, {y}]")
"""
from skopt import gp_minimize

# 定義する目的関数
def objective_function(params):
    x, y = params  # パラメータを取り出す
    x = float(x)
    y = float(y)
    return (x - 3)**2 + (y - 4)**2  # 仮の目的関数

# 変数の範囲を指定
dimensions = [["1","2","3","4","5"], ["1","2","3","4","5"]]  # 2つの変数が [1, 2, 3, 4, 5] のいずれかの値を取る

# gp_minimizeを実行
result = gp_minimize(objective_function, dimensions=dimensions, verbose=True)

# 結果を表示
print("Best parameters:", result.x)
print("Minimum value:", result.fun)
"""
# from week6_haruo import read_input, cal_dist, calculate_total_distance, format_tour
# cities = read_input('input/input_1.csv')
# dist = cal_dist(cities) # 全てのエッジの距離が入った二次元配列
# tours = list(all_tour(8,8))
# res = []
# for tour in tours:
#     modified_tour = [x - 1 for x in tour]
#     # print(modified_tour,calculate_total_distance(dist,modified_tour))
#     res.append([calculate_total_distance(dist,modified_tour), modified_tour])
# res.sort()
# print(res)
# res_tour = [[0, 3, 1, 2, 4]]
# with open(f'../GoogleTSP/google-step-tsp/output_0.csv', 'w') as f:
#     f.write(format_tour(res_tour) + '\n')

# def create_3d_array(n):
#     # 三次元配列を生成
#     array_3d = [[(i, j) for j in range(n)] for i in range(n)]
#     return array_3d

# # 例として n = 3 を指定
# n = 3
# result = create_3d_array(n)

# print(result)
# 結果の表示
# for row in result:
#     print(row)

"""
def process_3d_array(input_array):
    co = input_array.copy()
    # タプルの数が同じである場合、同じ数を持つタプルを削除し、各行を変換
    for i in range(len(co)-1):
        flag = 0
        for j in range(len(co[i]) - 1, -1, -1):
            print(co[i][j])
            if co[i][j][0] == co[i][j][1]:
                co[i][j] = co[i+1][j]
                flag = 1
            elif flag == 1:
                co[i][j] = co[i+1][j]
    return co[:-1]

    # # 同じ数を持つタプルを削除
    # merged_array = [list(set(row)) for row in input_array]

    # return merged_array

# 例として入力配列を指定
input_array = [[(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)]]

# 関数を呼び出して結果を表示
result = process_3d_array(input_array)
for row in result:
    print(row)
"""

# def second_process(input_array):
#     co = input_array.copy()
#     # タプルの数が同じである場合、同じ数を持つタプルを削除し、各行を変換
#     for i in range(len(co)):
#         # flag = 0
#         for j in range(len(co[i])-1):
#             print(i,j)
#             if co[i][j][0] - co[i][j][1] == 1:
#                 co[i].remove(co[i][j])
#     return co

# print(second_process([[(1, 0), (0, 1), (0, 2)],[(2, 0), (2, 1), (1, 2)]]))

"""
import numpy as np
from quasimc.sobol import Sobol
import matplotlib.pyplot as plt

def plot_sobol_points(num_points=1000):
    # 二次元平面にSobol乱数を生成
    # 引数は 次元数、 シード(任意)
    sobol = Sobol(2, 5)
    sobol_points = sobol.generate(10000)
    # sobol_points = sobol_seq.i4_sobol_generate(2, num_points).T

    # プロット
    plt.scatter(sobol_points[0], sobol_points[1], s=5, alpha=0.5)
    plt.title(f'Sobol Points in 2D Space ({num_points} points)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

# 関数の呼び出し
plot_sobol_points()
"""

# a = [52, 49, 43, 55, 19, 61, 63, 1, 6, 9, 45, 41, 11, 23, 5, 29, 40, 48, 18, 37, 20, 26, 58, 47, 24, 59, 46, 22, 31, 3, 54, 27, 7, 33, 42, 39, 60, 8, 0, 50, 36, 10, 21, 28, 56, 13, 62, 4, 17, 15, 12, 16, 30, 57, 44, 25, 14, 32, 53, 38, 35, 51, 34, 2]
# b = [52, 49, 43, 55, 19, 61, 63, 1, 6, 9, 45, 41, 11, 23, 5, 29, 40, 48, 18, 37, 20, 26, 58, 47, 24, 59, 46, 22, 31, 3, 54, 27, 7, 33, 42, 39, 60, 8, 0, 50, 36, 10, 21, 28, 56, 13, 62, 4, 17, 15, 12, 16, 30, 57, 44, 25, 14, 32, 53, 38, 35, 51, 34, 2]
# print(a==b)

"""
from quasimc.sobol import Sobol
def plot_sobol_points(num_points):
    # 二次元平面にSobol乱数を生成
    # 引数は 次元数、 シード(任意)
    sobol = Sobol(2, scrumbled=True)
    sobol_points = sobol.generate(num_points)
    return sobol_points.T

print(plot_sobol_points(100))
"""
# from quasimc.sobol import Sobol
# def plot_sobol_points(num_points, box_range:list):
#         # 二次元平面にSobol乱数を生成
#         # 引数は 次元数、 シード(任意)
#         sobol = Sobol(2)
#         sobol_points = sobol.generate(num_points)
#         # box_rangeに合うようにsobol_pointsを線形変換
#         sobol_points[0] = sobol_points[0] * (box_range[0][1] - box_range[0][0]) + box_range[0][0]
#         sobol_points[1] = sobol_points[1] * (box_range[1][1] - box_range[1][0]) + box_range[1][0]
#         return sobol_points.T

# print(plot_sobol_points(100, [[0.45,0.5],[0.32,0.37]]))

"""
sese = set()
sese.add(1)
sese.add(5)
sese.add(3)
sese.add(3)
sese.add(2)
sese.add(2)
print(sese)
print(2 in sese)
print(sorted(list(sese)))
"""
















