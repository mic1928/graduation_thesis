#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:50:57 2023

@author: tomo.f
"""

import math
import heapq
# from common import read_input, format_tour
import time
import random
import re

def read_input(filename):
    with open(filename) as f:
        cities = []
        for line in f.readlines()[1:]:  # Ignore the first line.
            xy = line.split(',')
            cities.append((float(xy[0]), float(xy[1])))
        return cities

def format_tour(tour):
    return 'index\n' + '\n'.join(map(str, tour))

def distance(city1, city2):
    for i in range(len(city1)):
        if i == 0:
            dist = (city1[i] - city2[i]) ** 2
        else:
            dist += (city1[i] - city2[i]) ** 2
    return math.sqrt(dist)

# 全てのパスの距離を計算する
def cal_dist(cities):
    N = len(cities)
    dist = [[0] * N for i in range(N)]
    for i in range(N):
        for j in range(i, N):
            dist[i][j] = dist[j][i] = distance(cities[i], cities[j])
    return dist

def cal_shortpath(matrix):
    sorted_indices_matrix = []
    
    for row in matrix:
        # 行ごとに要素のインデックスを小さい順にソート
        sorted_indices = sorted(range(len(row)), key=lambda k: row[k])
        sorted_indices_matrix.append(sorted_indices[1:])
    
    return sorted_indices_matrix


# 最小全域木を張る関数
# プリム法
# dist: 隣接行列（重み付き）
# startpoint: 出発点
# N: 都市の数
# Output: 隣接リスト
def minimum_spanning_tree(dist, startpoint, N):
    graph = {} # 最小全域木の隣接リスト
    not_visited = set(range(N)) # 訪れていない都市をsetで管理
    heapq_list = [] # 訪れた都市から出ているエッジのうち、もっとも距離が小さいものを取り出すための優先度付きキュー(に変換前のもの)

    # 初期値に関して、graph, not_visited, heapq_list を更新する
    graph[startpoint] = []
    not_visited.remove(startpoint)
    for i in not_visited:
        # heapq_list: [(距離, 都市１, 都市２), ()...]
        heapq_list.append((dist[startpoint][i], startpoint, i))
    heapq.heapify(heapq_list) #優先度付きキューに変換

    # 全ての都市を訪れるまで以下を繰り返す
    while not_visited:
        (cost, city1, city2) = heapq.heappop(heapq_list) # 最も距離が短いエッジを取り出す
        # city1は訪問済み
        # city2の訪れているかもしれないし、訪れていないかもしれない
        # 両方探索済みの時はなにもしない

        if city2 in not_visited:
            not_visited.remove(city2)
            # (city2, city1)のエッジを隣接リストに保存, city2は初訪問
            graph[city2] = [city1]
            # (city1, city2)のエッジを隣接リストに保存, city1は訪問済み
            graph[city1].append(city2)
            # 優先度付きキューに新しく探索可能になったエッジを追加する
            for i in not_visited:
                heapq.heappush(heapq_list, (dist[city2][i], city2, i))
        
    return graph

def tsp_approximation(graph, start_node:int):
    def dfs(node, visited):
        visited[node] = True
        tour.append(node)
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor, visited)
    visited = {node: False for node in graph}
    tour = []

    dfs(start_node, visited)

    # 最後にスタートノードに戻る
    tour.append(start_node)

    return tour



# 2-opt法
def two_opt_update(tour, dist, short_path):
    # print(short_path)
    # print(tour)
    tour.append(tour[0])
    N = len(tour)
    changed = True

    while changed:
        changed = False
    # 任意の二つのエッジを選択
        for i in range(N-3):
            # print(short_path)
            spot1 = tour[i]
            spot1_end = tour[i + 1]
            # print(f"tour:{tour},spot1:{spot1},spot1_end:{spot1_end},short_path[spot1]:{short_path[spot1]}")
            # print("short_path[spot1]:",short_path[spot1])
            index_spot = short_path[spot1].index(spot1_end)
            j_array = short_path[spot1][:index_spot]
            for i_ in range(i+2):
                if tour[i_] in j_array:
                    j_array.remove(tour[i_])
            for j in j_array:
                spot2 = j
                j = tour.index(j)
                spot2_end = tour[j + 1]
                if (dist[spot1][spot1_end] + dist[spot2][spot2_end]) > (dist[spot1][spot2] + dist[spot1_end][spot2_end]):
                    tour = tour[:i+1] + list(reversed(tour[i+1:j+1])) + tour[j+1:]
                    changed = True
                    break
            if changed == True:
                break
    tour.pop(-1)
    return tour

# 2-opt法
def two_opt_classic(tour, dist):
    tour.append(tour[0])
    N = len(tour)
    changed = True

    while changed:
        changed = False
    # 任意の二つのエッジを選択
        for i in range(N-3):
            spot1 = tour[i]
            spot1_end = tour[i + 1]
            for j in range(i+2, N-1):
                spot2 = tour[j]
                spot2_end = tour[j + 1]
                if (dist[spot1][spot1_end] + dist[spot2][spot2_end]) > (dist[spot1][spot2] + dist[spot1_end][spot2_end]):
                    tour = tour[:i+1] + list(reversed(tour[i+1:j+1])) + tour[j+1:]
                    changed = True
                    break
            if changed == True:
                break
    tour.pop(-1)
    return tour


def optimal_tour(distance_matrix, startpoint:int, short_path:list, is_atsp:bool=False):
    N = len(distance_matrix)
    if startpoint is None:
        random.seed(42)  # ランダムシードを42に設定
        startpoint = random.randint(0, N - 1)  # 0からN-1までの整数を生成
    
    spanning_tree = minimum_spanning_tree(distance_matrix, startpoint, N) #最小全域木を生成
    two_ap_tour = tsp_approximation(spanning_tree, startpoint) #2近似アルゴリズム
    # lap_time = time.time()
    # print(f"2近似アルゴリズム計算時間:{lap_time-start_time}")
    two_opt_tour = two_ap_tour
    if not is_atsp:
        # two_opt_tour = two_opt_update(two_ap_tour, distance_matrix, short_path)   #2-optアルゴリズム
        two_opt_tour = two_opt_classic(two_ap_tour, distance_matrix)   #2-optアルゴリズム
    return two_opt_tour
    # return two_ap_tour

def calculate_total_distance(distance_matrix, tour):
    # print("最初のtour:",tour)
    tour_copy = tour.copy()
    if tour_copy[0] == tour_copy[-1]:
        # tour_copy.append(tour_copy[0])
        tour_copy = tour_copy[:-1]
    total_distance = 0.0
    num_cities = len(tour_copy)

    for i in range(num_cities - 1):
        from_city = tour_copy[i]
        to_city = tour_copy[i + 1]
        # print(f"from_city:{from_city},to_city:{to_city}")
        total_distance += distance_matrix[from_city][to_city]
    # 最後の都市から始点に戻る距離を追加
    total_distance += distance_matrix[tour_copy[-1]][tour_copy[0]]

    return round(total_distance,5)


def read_atsp_file(file_path):
    numbers = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Find the start of the data section
        data_start_index = lines.index("EDGE_WEIGHT_SECTION\n") + 1

        # Read data and convert to a list
        for i in range(data_start_index, len(lines) - 1):
            row = list(map(float, lines[i].split()))
            numbers.extend(row)
    
    with open(file_path, 'r') as file:
        content = file.read()
    # 正規表現を使用してDIMENSIONの後の数字を抽出
    match = re.search(r'DIMENSION:\s*(\d+)', content)
    if match:
        dimension = int(match.group(1))
    # Convert to 2D array
    result_2d_array = [numbers[i:i+dimension] for i in range(0, len(numbers), dimension)]

    return result_2d_array

def read_tsp_file(file_path):
    numbers = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Find the start of the data section
        data_start_index = lines.index("NODE_COORD_SECTION\n") + 1

        # Read data and convert to a list
        for i in range(data_start_index, len(lines) - 1):
            row = list(map(float, lines[i].split()))
            numbers.extend(row)
    
    with open(file_path, 'r') as file:
        content = file.read()
    # 正規表現を使用してDIMENSIONの後の数字を抽出
    match = re.search(r'DIMENSION : \s*(\d+)', content)
    if match:
        dimension = int(match.group(1))
    # Convert to 2D array
    result_2d_array = [numbers[i:i+3] for i in range(0, len(numbers), 3)]
    result_2d_array = [row[1:] for row in result_2d_array]
    assert len(result_2d_array) == dimension
    assert len(result_2d_array[0]) == 2

    return result_2d_array

if __name__ == '__main__':
    start_time = time.time()
    for i in [7]:
        """
        cities = read_input(f'input_{i}.csv')
        dist = cal_dist(cities) # 全てのエッジの距離が入った二次元配列
        N = len(cities)
        
        

        startpoint = 21 # Week5での最短経路の初期値
        spanning_tree = minimum_spanning_tree(dist, startpoint, N)
        # startpoint = 2
        
        tour = tsp_approximation(spanning_tree)
        print("tourの最初",tour[:10])
        
        new_tour = two_opt(tour, dist)
        # new_tour = tour
        print("new_tourの最初",new_tour[:10])

        with open(f'../GoogleTSP/google-step-tsp/output_{i}.csv', 'w') as f:
            f.write(format_tour(new_tour) + '\n')
        print("finished")
        """

        cities = read_input(f'input/input_{i}.csv')
        dist = cal_dist(cities) # 全てのエッジの距離が入った二次元配列
        # print("dist計算完了")
        lap_time = time.time()
        print(f"dist計算時間:{lap_time-start_time}")

        short_path = cal_shortpath(dist)
        lap_time = time.time()
        print(f"short_path計算時間:{lap_time-start_time}")

        res_tour = optimal_tour(dist, cities, 2)
        # print(res_tour)
        distance = calculate_total_distance(dist,res_tour)
        print(f"総距離：{distance}")
        with open(f'../GoogleTSP/google-step-tsp/output_{i}.csv', 'w') as f:
            f.write(format_tour(res_tour) + '\n')
    end_time = time.time()
    print(f"かかった時間:{end_time-start_time}")
