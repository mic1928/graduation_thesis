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
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

# 全てのパスの距離を計算する
def cal_dist(cities):
    N = len(cities)
    dist = [[0] * N for i in range(N)]
    for i in range(N):
        for j in range(i, N):
            dist[i][j] = dist[j][i] = distance(cities[i], cities[j])
    return dist


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
def two_opt(tour, dist):
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



def optimal_tour(distance_matrix, cities:list, startpoint:int=None):
    N = len(cities)
    if startpoint is None:
        random.seed(42)  # ランダムシードを42に設定
        startpoint = random.randint(0, N - 1)  # 0からN-1までの整数を生成
    
    spanning_tree = minimum_spanning_tree(distance_matrix, startpoint, N) #最小全域木を生成
    two_ap_tour = tsp_approximation(spanning_tree, startpoint) #2近似アルゴリズム
    two_opt_tour = two_opt(two_ap_tour, distance_matrix)   #2-optアルゴリズム
    return two_opt_tour

def calculate_total_distance(distance_matrix, tour):
    total_distance = 0.0
    num_cities = len(tour)

    for i in range(num_cities - 1):
        from_city = tour[i]
        to_city = tour[i + 1]
        total_distance += distance_matrix[from_city][to_city]

    # 最後の都市から始点に戻る距離を追加
    total_distance += distance_matrix[tour[-1]][tour[0]]

    return total_distance



if __name__ == '__main__':
    start_time = time.time()
    for i in [5]:
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
        res_tour = optimal_tour(dist, cities, 2)
        print(res_tour)
        distance = calculate_total_distance(dist,res_tour)
        print(f"総距離：{distance}")
        with open(f'../GoogleTSP/google-step-tsp/output_{i}.csv', 'w') as f:
            f.write(format_tour(res_tour) + '\n')
    end_time = time.time()
    print(f"かかった時間:{end_time-start_time}")
