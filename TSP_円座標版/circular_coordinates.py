from week6_haruo import read_input, cal_dist, optimal_tour, calculate_total_distance, format_tour
import time
import math
import random
import numpy as np
from skopt import gp_minimize
import matplotlib.pyplot as plt
# from skopt.plots import plot_convergence
from scipy.interpolate import griddata




def calculate_baseline_tour(cities, dist):
    # res_tour = optimal_tour(dist, cities,21)
    res_tour = optimal_tour(dist, cities,2)
    distance = calculate_total_distance(dist,res_tour)
    # with open(f'../GoogleTSP/google-step-tsp/output_{input_number}.csv', 'w') as f:
    #     f.write(format_tour(res_tour) + '\n')
    return res_tour,distance

def insert_at_position(lst, position):  #positionは0~都市数-3
    # 先頭要素を取得
    first_element = lst[0]
    # 先頭要素をリストから削除
    lst.pop(0)
    # 指定された位置に先頭要素を挿入
    lst.insert(position+1, first_element)
    return lst

def find_position(cumulative_probabilities, random_number):
    for i, probability in enumerate(cumulative_probabilities):
        if random_number >= probability and random_number <= cumulative_probabilities[i + 1]:
            return i

    # random_numberがリストの範囲を超える場合、最後の要素のインデックスを返す
    return len(cumulative_probabilities) - 1
    
#深さ1,深さ2,深さ3,...を指数関数に従って選ぶ
def distance_to_searchdepth(random_number:float, num_cities:int, max_searchdepth:int=3)->int:
    # 区間ごとの確率を計算
    probabilities = [(num_cities-2)**i / (num_cities-2)**max_searchdepth for i in range(max_searchdepth+1)]
    probabilities.insert(0,0)
    # print(probabilities)
    # 区間の選択
    selected_interval = find_position(probabilities, random_number)

    return selected_interval


#深さ1,深さ2,深さ3,...を等確率で選ぶ
def random_number_to_depth_equal_probabilities(random_number:float, max_searchdepth:int):
    interval_size = 1 / max_searchdepth
    result_intervals = [i * interval_size for i in range(max_searchdepth + 1)]
    position = find_position(result_intervals, random_number)
    return position


def angle_to_sector(angle:float, divisions:int, search_depth:int):
    one_sector_angle = math.pi * 2 / divisions**search_depth
    sector = math.floor(angle / one_sector_angle)
    return sector

def to_base_n(number, n, depth):
    """
    10進数のnumberをn進数に変換する
    
    Args:
        number (int): 10進数の数
        n (int): 変換後の進数
        depth (int): 変換後の桁数
    """
    if n < 2:
        raise ValueError("nは2以上の整数である必要があります")

    remainder = [number % n]
    for i in range(depth):
        number //= n
        remainder.append(number % n)
    return remainder[:-1][::-1]

def generate_swapped_route(base_route:list, angle:float, num_cities:int, search_depth:int):
    base_route_ = base_route.copy()
    if search_depth == 0:
        return base_route_
    divisions = num_cities-2
    sector_in_all = angle_to_sector(angle, divisions, search_depth)
    swap_order = to_base_n(sector_in_all, divisions, search_depth)
    print(f"swap_order:{swap_order}")
    for swap_position in swap_order:
        base_route_ = insert_at_position(base_route_, swap_position)
    return base_route_

def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def plot_heatmap(coordinates, values):
    # Extract x and y coordinates from the list of 2D coordinates
    for i in range(len(coordinates)):
        coordinates[i] = polar_to_cartesian(coordinates[i][0], coordinates[i][1])

    x_coords, y_coords = zip(*coordinates)

    # Define a grid for the heatmap
    x_grid, y_grid = np.mgrid[min(x_coords):max(x_coords):500j, min(y_coords):max(y_coords):500j]
    # x_grid, y_grid = np.mgrid[-1:1:500j, -1:1:500j]

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

def bayse_tsp(param = None):
    random_distance = param[0]
    random_angle = param[1]
    # print(f"random_distance:{random_distance}, random_angle:{random_angle}")

    # search_depth = distance_to_searchdepth(random_distance, num_cities,10)
    search_depth = random_number_to_depth_equal_probabilities(random_distance, 10)
    print(f"search_depth:{search_depth},random_distance:{random_distance}, random_angle:{random_angle}")
    new_route = generate_swapped_route(baseline_tour, random_angle, num_cities, search_depth+1)
    # print(f"new_route:{new_route[:5]}, baseline_tour:{baseline_tour[:5]}")

    new_distance = calculate_total_distance(dist,new_route)
    print(f"new_distance:{new_distance}, distance:{distance}")

    return new_distance

if __name__ == '__main__':
    cities = read_input('input/input_2.csv')
    dist = cal_dist(cities) # 全てのエッジの距離が入った二次元配列

    baseline_tour,distance = calculate_baseline_tour(cities, dist)
    # random.shuffle(baseline_tour)
    print(f"総距離：{distance}")
    num_cities = len(baseline_tour)-1
    print(f"都市数：{num_cities}")

    x1 = np.array([0.0, 1.0])
    x2 = np.array([0.01, math.pi*2-0.01])
    # x2 = np.array([0.01, 0.02])
    x = (x1, x2)
    result = gp_minimize(bayse_tsp, x, 
                          acq_func="EI",
                          n_calls=100,
                          n_initial_points = 100,
                          noise=0.0,
                          model_queue_size=1,
                          verbose=True,
                        #   xi=0.00001,
                        #   callback=plot_heatmap,
                        #   n_points=100
                        )
    
    # Plot the convergence plot
    # plot_convergence(result)
    # plt.show()

    print(result)
    print(f"元の経路の距離：{distance}")
    print(f"最適な経路の距離：{result.fun}")
    coordinates = result.x_iters
    values = result.func_vals
    plot_heatmap(coordinates, values)

    # print(bayse_tsp([0.0, 1.0398220004185816]))
    # print(generate_swapped_route(baseline_tour, 0.0, num_cities, 1))
    # print(generate_swapped_route([1,2,3,4,5], 0.0, 512, 1))


