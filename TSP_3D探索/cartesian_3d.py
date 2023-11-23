from week6_haruo_3d import read_input, cal_dist, optimal_tour, calculate_total_distance, format_tour, distance, cal_shortpath

# from solver_greedy import solve
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from quasimc.sobol import Sobol

class Baseline_first:
    def __init__(self, file_number:int, dist:list, short_path:list, first_point:int=0 ,baseline_tour:list=None):
        self.file_number = file_number
        self.first_point = first_point
        self.dist = dist
        self.short_path = short_path
        self.num_cities = len(self.dist)
        self.calculate_baseline_tour()

    def calculate_baseline_tour(self):
        res_tour = optimal_tour(self.dist, self.short_path, self.first_point)
        distance = calculate_total_distance(self.dist,res_tour)
        self.baseline_tour = res_tour
        self.distance = distance
    
class Coordinate:
    def __init__(self, N:int):
        self.N = N
        self.array_3d = self.integrate_process()

    def create_3d_array(self):
        n = self.N
        array_3d = [[(i, j) for j in range(n)] for i in range(n)]
        return array_3d
    
    def first_process(self, input_array):   #同じ要素を持つタプルを削除
        co = input_array.copy()
        for i in range(len(co)-1):
            flag = 0
            for j in range(len(co[i]) - 1, -1, -1):
                if co[i][j][0] == co[i][j][1]:
                    co[i][j] = co[i+1][j]
                    flag = 1
                elif flag == 1:
                    co[i][j] = co[i+1][j]
        return co[:-1]
    
    def second_process(self, input_array):  #(1,0)と(0,1)は等価なので前者を削除
        co = input_array.copy()
        for i in range(len(co)):
            for j in range(len(co[i])-1):
                if co[i][j][0] - co[i][j][1] == 1:
                    co[i].remove(co[i][j])
        return co

    def integrate_process(self):
        array1 = self.create_3d_array()
        array2 = self.first_process(array1)
        array3 = self.second_process(array2)
        return array3

class Swap:
    def __init__(self, array_3d:list, baseline_tour:list, dist:list):
        self.baseline_tour = baseline_tour
        self.array_3d = array_3d
        self.dist = dist

    def insert_at_position(self, lst:list, remove_position:int, insert_position:int)->list:
        lst_copy = lst.copy()
        assert remove_position != insert_position
        # 先頭要素を取得
        first_element = lst_copy[remove_position]
        # 先頭要素をリストから削除
        lst_copy.pop(remove_position)
        # 指定された位置に先頭要素を挿入
        lst_copy.insert(insert_position, first_element)
        return lst_copy
    
    def assign_sector(self, N, float1, float2):
        width = 1 / N
        if float1 == 1:
            float1 -= 0.0000001
        if float2 == 1:
            float2 -= 0.0000001
        i = math.floor(float1 / width)  #y座標
        j = math.floor(float2 / width)  #x座標
        # print(f"float1:{float1}, float2:{float2},i:{i}, j:{j}")
        return i, j

    def swap_order(self, float1, float2):
        array_3d = self.array_3d.copy()
        N = len(array_3d)
        i, j = self.assign_sector(N, float1, float2)
        remove_pos, insert_pos = array_3d[i][j][0], array_3d[i][j][1]
        # print(f"remove_pos:{remove_pos}, insert_pos:{insert_pos}")
        return remove_pos, insert_pos

    def swap_and_distance(self, float1, float2):
        if self.baseline_tour[0] == self.baseline_tour[-1]:
            baseline_tour_copy = self.baseline_tour.copy()[:-1]
        else:
            baseline_tour_copy = self.baseline_tour.copy()
        remove_pos, insert_pos = self.swap_order(float1, float2)
        new_tour = self.insert_at_position(baseline_tour_copy, remove_pos, insert_pos)
        new_distance = calculate_total_distance(self.dist, new_tour)
        return new_tour, new_distance

class Tour:
    def __init__(self, length:int, order:list, random_number:list, box_order:float ,already_baseline:bool=False):
        self.length = length
        self.order = order
        self.random_number = random_number
        self.already_baseline = already_baseline
        self.box_order = box_order
        # self.next_box_width = 10**(-(box_order+1))

    def box(self):
        self.box_width = 10**(-self.box_order)
        horizontal_under = self.random_number[0] - self.box_width/2
        horizontal_upper = self.random_number[0] + self.box_width/2
        vertical_under = self.random_number[1] - self.box_width/2
        vertical_upper = self.random_number[1] + self.box_width/2
        if horizontal_under < 0:
            horizontal_under = 0
        if horizontal_upper > 1:
            horizontal_upper = 1
        if vertical_under < 0:
            vertical_under = 0
        if vertical_upper > 1:
            vertical_upper = 1
        return [[horizontal_under, horizontal_upper], [vertical_under, vertical_upper]]

class Search_in_same_baseline:
    def __init__(self, baseline_tour:Tour):
        self.baseline_tour = baseline_tour
        self.baseline_tour.already_baseline = True
        already_baseline_length.add(self.baseline_tour.length)
        self.baseline_tour.box_order = 0
        self.baseline_order = baseline_tour.order
        self.baseline_order_length = calculate_total_distance(dist, self.baseline_order)
        self.N = len(set(self.baseline_order))
        self.array_3d = self.create_3d_array()
        self.swap = Swap(self.array_3d, self.baseline_order, dist)
        # self.from_top = self.search_all(self.baseline_tour)
        self.search_times = 100

    def create_3d_array(self):
        return Coordinate(self.N).array_3d
    
    def plot_sobol_points(self, num_points, box_range:list):
        # 二次元平面にSobol乱数を生成
        # 引数は 次元数、 シード(任意)
        sobol = Sobol(2)
        sobol_points = sobol.generate(num_points)
        # box_rangeに合うようにsobol_pointsを線形変換
        sobol_points[0] = sobol_points[0] * (box_range[0][1] - box_range[0][0]) + box_range[0][0]
        sobol_points[1] = sobol_points[1] * (box_range[1][1] - box_range[1][0]) + box_range[1][0]
        return sobol_points.T
    
    def random_number_in_box(self, box_range:list):
        random1 = random.uniform(box_range[0][0], box_range[0][1])
        random2 = random.uniform(box_range[1][0], box_range[1][1])
        return random1, random2
    
    def search(self, center_tour:Tour, use_sobol:bool=False):
        tours = [center_tour]
        box_order = center_tour.box_order
        box_range = center_tour.box()
        # if self.baseline_tour in tours:
        #     tours.remove(self.baseline_tour)
        # print(box_range)
        if use_sobol:
            sobol_array = self.plot_sobol_points(self.search_times, box_range)
        for i in range(self.search_times):
            if use_sobol:
                random1,random2 = sobol_array[i]
            else:
                random1,random2 = self.random_number_in_box(box_range)
            
            new_order, new_distance = self.swap.swap_and_distance(random1, random2)
            if new_distance == self.baseline_order_length:
                new_tour = Tour(new_distance, new_order, [random1, random2], box_order+1, True)
            else:
                new_tour = Tour(new_distance, new_order, [random1, random2], box_order+1)
            if (all(tour.length != new_tour.length for tour in tours) or 
            # all(distance(tour.random_number, new_tour.random_number) > 10**(-box_order-1) for tour in tours)):
            all(distance(tour.random_number, new_tour.random_number) > 0.1 for tour in tours)):
                    tours.append(new_tour)
            # length 属性でソート
            tours = sorted(tours, key=lambda x: x.length)
        center_tour.box_order += 1
        if self.baseline_tour in tours:
            tours.remove(self.baseline_tour)
        return tours[:3]

    def search_all(self):
        self.baseline_tour.random_number = [0.5, 0.5]
        tours_all = []

        new_order, new_distance = self.swap.swap_and_distance(0.0,1.0)
        new_tour = Tour(new_distance, new_order, [0.0,1.0], 0, True)
        tours_all.append(new_tour)
        new_order, new_distance = self.swap.swap_and_distance(1.0,0.0)
        new_tour = Tour(new_distance, new_order, [1.0,0.0], 0, True)
        tours_all.append(new_tour)

        tours = self.search(self.baseline_tour)
        for tour in tours:
            if (all(tour_all.length != tour.length for tour_all in tours_all) or 
                all(distance(tour.random_number, tour_all.random_number) > 0.1 for tour_all in tours)):
                    tours_all.append(tour)
        tours_all = sorted(tours_all, key=lambda x: x.length)

        # for i in range(min(self.N//3,50)):
        for i in range(min(self.N,10)):
            min_order_tour = sorted(tours_all, key=lambda x: x.box_order)[0]
            tours = self.search(min_order_tour)
            for tour in tours:
                if (all(tour_all.length != tour.length for tour_all in tours_all) or 
                all(distance(tour.random_number, tour_all.random_number) > 0.1 for tour_all in tours)):
                    tours_all.append(tour)
            tours_all = sorted(tours_all, key=lambda x: x.length)[:3]
            # print(f"i:{i:>3}, length:{tours_all[0].length}, box_order:{tours_all[0].box_order}")
        # if self.baseline_tour in tours_all:
        #     tours_all.remove(self.baseline_tour)
        return tours_all

class Search_in_different_baseline:
    def __init__(self, first_baseline_tour:Tour):
        self.first_baseline_tour = first_baseline_tour
        self.first_baseline_tour.already_baseline = True
        self.first_baseline_tour.box_order = 0
    
    def search(self, tours_all:list):
        # tours_all = [self.first_baseline_tour]
        tours_all_copy = tours_all.copy()
        for i, tour in enumerate(tours_all_copy):
            if tour.already_baseline == True:
                continue
            if tour.length in already_baseline_length:
                continue
            search_obj = Search_in_same_baseline(tour)
            tours = search_obj.search_all()
            for tour in tours:
                if all(tour_all.length != tour.length for tour_all in tours_all):
                    tours_all_copy.append(tour)
            tours_all_copy = sorted(tours_all_copy, key=lambda x: x.length)[:5]
            # print(f"i:{i:>2}番目, length:{tours_all_copy[0].length}")
        return sorted(tours_all_copy, key=lambda x: x.length)[:5]
    
    def search_all(self):
        self.first_baseline_tour.already_baseline = False
        tours_all = [self.first_baseline_tour]
        last_length = 0
        last_2_length = 0
        for i in range(100):
            # print(len(tours_all))
            tours = self.search(tours_all)
            for tour in tours:
                if all(tour_all.length != tour.length for tour_all in tours_all):
                    # if tour.length in already_baseline_length:
                    #     continue
                    tours_all.append(tour)
            # print(len(tours_all))
            tours_all = sorted(tours_all, key=lambda x: x.length)[:5]
            print(f"i:{i:>2}, length:{tours_all[0].length:<12}, tours_all_len:{len(tours_all)},\
            num_True:{sum(tour_all.already_baseline == True for tour_all in tours_all)}\
            num_already_baseline:{sum(tour_all.length in already_baseline_length for tour_all in tours_all)}")
            
            if all(tour_all.already_baseline == True for tour_all in tours_all):
                break
            if all(tour_all.length in already_baseline_length for tour_all in tours_all):
                break
            if last_2_length == tours_all[0].length:
                break
            last_2_length = last_length
            last_length = tours_all[0].length
        return tours_all
        
class Different_first_baseline:
    def __init__(self, file_number:int, dist:list, short_path:list):
        self.file_number = file_number
        cities = read_input(f'input/input_{self.file_number}.csv')
        self.N = len(cities)
        self.dist = dist
        self.short_path = short_path
        print(f"都市数：{self.N}だよ")

    def search(self):
        tours_all = []
        start_time = time.time()
        search_city_num = self.N//10
        for start in range(search_city_num):
            baseline = Baseline_first(self.file_number, self.dist, self.short_path, start)
            first_baseline_tour = Tour(baseline.distance, baseline.baseline_tour, [0.5, 0.5], 0)
            search_obj = Search_in_different_baseline(first_baseline_tour)
            tours = search_obj.search_all()
            for tour in tours:
                if all(tour_all.length != tour.length for tour_all in tours_all):
                    tours_all.append(tour)
            tours_all = sorted(tours_all, key=lambda x: x.length)[:3]
            print(f"start:{start:>3}, これまでの最短経路長:{tours_all[0].length}")
            end_time = time.time()
            # print(f"探索済みの経路長：{sorted(list(already_baseline_length))}")
            print(f"探索済みの経路長：{len(already_baseline_length)}個")
            print(f"経過時間：{end_time - start_time:.1f}秒, 予想残り時間：{((end_time - start_time)/(start+1)) *(search_city_num-start-1):.1f}秒")
        return tours_all

# 近い都市をクラスタリングでまとめる
# 突然変異を起こす
# 円環座標で探索する

if __name__ == '__main__':
    file_num = 4
    cities = read_input(f'input/input_{file_num}.csv')
    dist = cal_dist(cities) # 全てのエッジの距離が入った二次元配列
    short_path = cal_shortpath(dist)
    search_times = 100
    already_baseline_length = set()

    dodo = Different_first_baseline(file_num, dist, short_path)
    top3 = dodo.search()
    last_1 = Search_in_same_baseline(top3[0]).search_all()[0]
    top1_length = last_1.length
    print(f"最短経路は...:{top1_length}")
    for tour in top3:
        print(tour.length)
    # top3.sort(key=lambda x: x.length)
    # top_order = top3.copy()[0].order
    with open(f'../GoogleTSP/google-step-tsp/output_{file_num}.csv', 'w') as f:
        f.write(format_tour(last_1.order) + '\n')

"""
    x1 = np.array([0.0, 1.0])
    x2 = np.array([0.0, 1.0])
    x = (x1, x2)
    result = gp_minimize(bayse_tsp, x, 
                          acq_func="EI",
                          n_calls=310,
                          n_initial_points = 300,
                          noise=0.0,
                          model_queue_size=1,
                          verbose=True,
                        #   xi=0.000000001,
                        #   callback=plot_heatmap,
                          n_points=75
                        )

    print(result)
    print(f"元の経路の距離：{distance}")
    print(f"最適な経路の距離：{result.fun}")
    # print(f"最適なx:{result.x}")
    coordinates = result.x_iters
    values = result.func_vals
    plot_heatmap(coordinates, values, num_cities)
"""
