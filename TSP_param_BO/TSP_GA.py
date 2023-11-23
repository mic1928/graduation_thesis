#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 23:19:11 2023

@author: tomo.f
"""

import random

# 都市の座標を定義
"""
cities = {
    'A': (0, 0),
    'B': (1, 3),
    'C': (2, 5),
    'D': (5, 2),
    'E': (6, 0)
}
"""
def generate_cities():
    # 空の辞書を作成
    cities = {}

    # 1から30までの整数をキーとし、1から100までのランダムな整数のタプルを値として辞書に追加
    for key in range(1, 31):
        value = (random.randint(1, 100), random.randint(1, 100))
        cities[str(key)] = value
        
    return cities


# 初期個体群の生成
def generate_individual(cities):
    cities_list = list(cities)
    return random.sample(cities_list, len(cities_list))

# 初期個体群の評価
def evaluate_individual(individual, cities):
    total_distance = 0
    for i in range(len(individual) - 1):
        city1 = individual[i]
        city2 = individual[i + 1]
        distance = calculate_distance(cities[city1], cities[city2])
        total_distance += distance
    # 最後の都市から最初の都市に戻る距離を追加
    total_distance += calculate_distance(cities[individual[-1]], cities[individual[0]])
    return total_distance

# 2つの都市間の距離を計算
def calculate_distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# 選択
def select(population, cities, num_parents):
    parents = []
    for _ in range(num_parents):
        random.shuffle(population)
        selected = population[:num_parents]
        best_individual = min(selected, key=lambda ind: evaluate_individual(ind, cities))
        parents.append(best_individual)
    return parents

# 交叉
def crossover(parents):
    child = [''] * len(parents[0])
    crossover_point = random.randint(1, len(parents[0]) - 1)
    child[:crossover_point] = parents[0][:crossover_point]
    for gene in parents[1]:
        if gene not in child:
            child[crossover_point] = gene
            crossover_point += 1
    return child

# 突然変異
def mutate(individual):
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

# 遺伝的アルゴリズムを実行
def genetic_algorithm(population_size = 100, num_generations = 317, mutate_threshold = 1.0):
    # int(params[0]):population_size, int(params[1]):num_generations, int(params[2]):mutate_threshold
    # print("ああああああああああああああああああああああああ",params)
    # cities = generate_cities()
    cities = {'1': (43, 93), '2': (57, 93), '3': (63, 2), '4': (96, 65), '5': (25, 4), '6': (99, 40), '7': (54, 28), '8': (60, 1), '9': (78, 65), '10': (45, 35), '11': (93, 86), '12': (24, 24), '13': (79, 53), '14': (56, 94), '15': (94, 38), '16': (41, 10), '17': (58, 70), '18': (47, 4), '19': (56, 21), '20': (61, 73), '21': (85, 72), '22': (88, 84), '23': (80, 95), '24': (81, 78), '25': (40, 31), '26': (91, 86), '27': (78, 49), '28': (49, 12), '29': (36, 36), '30': (25, 66)}
    population = [generate_individual(cities) for _ in range(population_size)]
    for generation in range(num_generations):
        if generation%100 == 0:
            print(f"generation:{generation}")
        parents = select(population, cities, population_size // 2)
        new_population = parents.copy()
        while len(new_population) < population_size:
            child = crossover(random.sample(parents, 2))
            if random.random() < mutate_threshold:
                mutate(child)
            new_population.append(child)
        population = new_population

    # 最適な解を返す
    best_individual = min(population, key=lambda ind: evaluate_individual(ind, cities))
    return best_individual, evaluate_individual(best_individual, cities)
    # return evaluate_individual(best_individual, cities)

if __name__ == '__main__':
    best_solution, best_distance = genetic_algorithm(population_size = 100, num_generations = 1000, mutate_threshold = 0.1)
    print("最適なルート:", best_solution)
    print("最短距離:", best_distance)
