import numpy as np
from skopt import gp_minimize

def func(param=None):
    ret = np.cos(param[0] + 2.34) + np.cos(param[1] - 0.78)
    print(f"param[0]:{param[0]}, param[1]:{int(param[1])}")
    return -ret

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


from TSP_GA import generate_cities
from TSP_GA import generate_individual
from TSP_GA import evaluate_individual
from TSP_GA import calculate_distance
from TSP_GA import select
from TSP_GA import crossover
from TSP_GA import mutate
import random
# from TSP_GA import genetic_algorithm

# 遺伝的アルゴリズムを実行
def genetic_algorithm(params = None):
    # int(params[0]):population_size, int(params[1]):num_generations, int(params[2]):mutate_threshold
    print("params:",params)
    # cities = generate_cities()
    # print(cities)
    cities = {'1': (43, 93), '2': (57, 93), '3': (63, 2), '4': (96, 65), '5': (25, 4), '6': (99, 40), '7': (54, 28), '8': (60, 1), '9': (78, 65), '10': (45, 35), '11': (93, 86), '12': (24, 24), '13': (79, 53), '14': (56, 94), '15': (94, 38), '16': (41, 10), '17': (58, 70), '18': (47, 4), '19': (56, 21), '20': (61, 73), '21': (85, 72), '22': (88, 84), '23': (80, 95), '24': (81, 78), '25': (40, 31), '26': (91, 86), '27': (78, 49), '28': (49, 12), '29': (36, 36), '30': (25, 66)}
    population = [generate_individual(cities) for _ in range(int(params[0]))]
    for generation in range(int(params[1])):
        if generation%100 == 0:
            print(f"generation:{generation}")
        parents = select(population, cities, int(params[0]) // 2)
        new_population = parents.copy()
        while len(new_population) < int(params[0]):
            child = crossover(random.sample(parents, 2))
            if random.random() < int(params[2]):
                mutate(child)
            new_population.append(child)
        population = new_population

    # 最適な解を返す
    best_individual = min(population, key=lambda ind: evaluate_individual(ind, cities))
    # return best_individual, evaluate_individual(best_individual, cities)
    return evaluate_individual(best_individual, cities)


# def genetic_algorithm(params = None):
#     params = [int(param) for param in params]  # パラメータを整数型に変換
#     return params[0]+params[1]-params[2]

if __name__ == '__main__':
    x1 = np.array([4.0, 100.0])
    x2 = np.array([1.0, 1000.0])
    x3 = np.array([0.1, 1.0])
    x = (x1, x2, x3)
    result = gp_minimize(genetic_algorithm, x, 
                          n_calls=100,
                          noise=0.0,
                          model_queue_size=1,
                          verbose=True)
    print(result)
    # best_solution, best_distance = genetic_algorithm(population_size=100, num_generations=1000)
    # print("最適なルート:", best_solution)
    # print("最短距離:", best_distance)
