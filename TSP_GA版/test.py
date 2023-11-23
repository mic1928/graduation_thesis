# def my_decorator(func):
#     def wrapper():
#         print("Something is happening before the function is called.")
#         func()
#         print("Something is happening after the function is called.")
#     return wrapper

# @my_decorator
# def say_hello():
#     print("Hello!")

# say_hello()

# import numpy as np
# from GaTsp import *

# loggers = (
#     Logger_trace(level=2),
#     Logger_leaders(),
#     Logger_fitness(),
#     Logger_population(),
#     Logger_population(show_breeded=True),
#     Logger_last_fitness_histgram(),
# )
# loggers = (Logger_trace(level=2), Logger_leaders())
# spc_r = Species(N=80, seed=30)
# mdl = Model(species=spc_r, max_population = 4000)
# mdl.fit(loggers=loggers)
# Logger.plot_all(loggers=loggers)






# if __name__ == '__main__':
#     x1 = (-np.pi, np.pi)
#     x2 = (-np.pi, np.pi)
#     x = (x1, x2)
#     result = gp_minimize(func, x, 
#                           n_calls=30,
#                           noise=0.0,
#                           model_queue_size=1,
#                           verbose=True)
#     print(result)


# external_var = 42

# class MyClass:
#     def access_external_var(self):
#         print("a")
#         global external_var
#         external_var = 43

# my_instance = MyClass()
# print(external_var)
# my_instance.access_external_var()  # グローバル変数へのアクセス
# print(external_var)


# import math
# N = 30
# nepochs = int(N * math.log(N)**2)
# print(nepochs*3)


"""
def tsp_approximation(graph):
    def dfs(node, visited):
        visited[node] = True
        tour.append(node)
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor, visited)

    start_node = list(graph.keys())[0]
    visited = {node: False for node in graph}
    tour = []

    dfs(start_node, visited)

    # 最後にスタートノードに戻る
    tour.append(start_node)

    return tour

# 入力例
input_graph = {
    2: [7],
    7: [2, 3, 0],
    3: [7, 5],
    5: [3, 1],
    1: [5, 6],
    6: [1],
    0: [7, 4],
    4: [0]
}

approx_tour = tsp_approximation(input_graph)
print(approx_tour)
"""

# def swap_adjacent_elements(input_list):
#     swaps = []

#     for i in range(len(input_list) - 1):
#         # 隣接した要素を入れ替え
#         swapped_list = input_list[:i] + [input_list[i + 1], input_list[i]] + input_list[i + 2:]
#         swaps.append(swapped_list)
    
#     swap_first_and_last = input_list.copy()
#     swap_first_and_last[0], swap_first_and_last[-1] = input_list[-1], input_list[0]  # 最初と最後の要素を入れ替え
#     swaps.append(swap_first_and_last)

#     return swaps


# input_list = [1, 2, 3, 4, 5,6,7,8,9,10]
# result = swap_adjacent_elements(input_list)
# unique_results = set(tuple(sublist) for sublist in result)
# print(set(unique_results))

"""
def calculate_total_distance(distance_matrix, tour):
    total_distance = 0.0
    num_cities = len(tour)

    for i in range(num_cities - 1):
        from_city = tour[i]
        to_city = tour[i + 1]
        print(f"from_city:{from_city}, to_city:{to_city}")
        total_distance += distance_matrix[from_city][to_city]

    # 最後の都市から始点に戻る距離を追加
    total_distance += distance_matrix[tour[-1]][tour[0]]

    return total_distance

# 入力例
distance_matrix = [
    [0.0, 1139.468611035281, 679.7227326641358, 829.251122595876, 740.0208580992705, 1182.6681474124582, 1260.2227387060204, 510.37258658240955],
    [1139.468611035281, 0.0, 463.63085520669887, 512.7321993957855, 1091.1135139211965, 197.09026839179194, 227.6792328924561, 660.1217509001684],
    [679.7227326641358, 463.63085520669887, 0.0, 394.51229505232465, 745.9866861116151, 544.7360119219416, 586.8664381481045, 257.6060664927319],
    [829.251122595876, 512.7321993957855, 394.51229505232465, 0.0, 1124.5662308439055, 435.2752536286654, 729.7707003574172, 335.42409263822515],
    [740.0208580992705, 1091.1135139211965, 745.9866861116151, 1124.5662308439055, 0.0, 1241.5976141587041, 1069.7335327655492, 839.9769883576002],
    [1182.6681474124582, 197.09026839179194, 544.7360119219416, 435.2752536286654, 1241.5976141587041, 0.0, 405.0357274497853, 676.5180960390303],
    [1260.2227387060204, 227.6792328924561, 586.8664381481045, 729.7707003574172, 1069.7335327655492, 405.0357274497853, 0.0, 822.3880322363773],
    [510.37258658240955, 660.1217509001684, 257.6060664927319, 335.42409263822515, 839.9769883576002, 676.5180960390303, 822.3880322363773, 0.0]
]
tour = [2, 6, 1, 5, 3, 7, 0, 4, 2]

total_distance = calculate_total_distance(distance_matrix, tour)
print("Total Distance:", total_distance)
"""

# print(set([[1, 0, 2, 3, 4, 5], [2, 1, 0, 3, 4, 5], [3, 1, 2, 0, 4, 5], [4, 1, 2, 3, 0, 5], [5, 1, 2, 3, 4, 0]]))

"""
import math

value = 9.304130799145518e-10

# 常用対数を取る
log_value = math.log10(value)

print(log_value)
"""

def generate_triangle_numbers(n):
    triangle_numbers = [sum(range(1, i + 1)) for i in range(1, n + 1)]
    return triangle_numbers

# 例: 要素数が5の場合
n = 5
tri_numbers = generate_triangle_numbers(n)
print(tri_numbers)


