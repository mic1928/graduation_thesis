from week6_3d import optimal_tour, calculate_total_distance, distance, read_atsp_file

# from solver_greedy import solve
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from quasimc.sobol import Sobol
from tqdm import tqdm
import qmcpy as qp

class Baseline_first:
    def __init__(self, dist:list, short_path:list, first_point:int=0 ,baseline_tour:list=None):
        self.first_point = first_point
        self.dist = dist
        self.short_path = short_path
        self.num_cities = len(self.dist)
        self.calculate_baseline_tour()

    def calculate_baseline_tour(self):
        res_tour = optimal_tour(self.dist, self.first_point, self.short_path, True)
        # res_tour = list(np.random.permutation(self.num_cities))
        distance = calculate_total_distance(self.dist,res_tour)
        print(f"経路長：{distance}")
        self.baseline_tour = res_tour
        self.distance = distance

class Tour:
    def __init__(self, length:float, order:list, random_number:list, box_order:int ,already_baseline:bool=False):
        self.length = length
        self.order = order
        self.random_number = random_number
        self.already_baseline = already_baseline
        self.box_order = box_order

    def box(self):
        self.box_width = 10**(-self.box_order)
        x_under = self.random_number[0] - self.box_width/2
        x_upper = self.random_number[0] + self.box_width/2
        y_under = self.random_number[1] - self.box_width/2
        y_upper = self.random_number[1] + self.box_width/2
        z_under = self.random_number[2] - self.box_width/2
        z_upper = self.random_number[2] + self.box_width/2
        x_under = 0 if x_under < 0 else x_under
        x_upper = 1 if x_upper > 1 else x_upper
        y_under = 0 if y_under < 0 else y_under
        y_upper = 1 if y_upper > 1 else y_upper
        z_under = 0 if z_under < 0 else z_under
        z_upper = 1 if z_upper > 1 else z_upper

        return [[x_under, x_upper], [y_under, y_upper],[z_under, z_upper]]

class Swap:
    def __init__(self, baseline_tour:Tour, dist:list):
        if baseline_tour.order[0] == baseline_tour.order[-1]:
            self.baseline_order = baseline_tour.order[:-1]
        else:
            self.baseline_order = baseline_tour.order
        self.baseline_length = baseline_tour.length
        # self.array_3d = array_3d
        self.dist = dist
        self.N = len(dist)

    def insert_at_position(self, lst_copy, remove_length, remove_position, insert_position):
        # 準備
        minus = False
        if remove_length < 0:
            minus = True
            remove_length = -remove_length
        turn = (remove_position + remove_length > self.N)
        # リストから取り除く要素を取得
        double_lst = lst_copy + lst_copy
        removed_elements = double_lst[remove_position : (remove_position + remove_length)]
        # リストから要素を取り除く
        if turn:
            lst_copy = lst_copy[(remove_position + remove_length - self.N):remove_position]
        else:
            lst_copy = lst_copy[:remove_position] + lst_copy[(remove_position + remove_length):]
        # リストに要素を挿入する
        insert_before = lst_copy[insert_position-1]
        insert_after = lst_copy[insert_position%len(lst_copy)]
        if minus:
            lst_copy = lst_copy[:insert_position] + removed_elements[::-1] + lst_copy[insert_position:]
            insert_first = removed_elements[-1]
            insert_last = removed_elements[0]
        else:
            lst_copy = lst_copy[:insert_position] + removed_elements + lst_copy[insert_position:]
            insert_first = removed_elements[0]
            insert_last = removed_elements[-1]
        return lst_copy, insert_first, insert_last, insert_before, insert_after
        
    def assign_sector(self, float1, float2, float3): #省略の工夫あり
        N = self.N

        float1 = (1 - 0.00000001) if float1 == 1 else float1
        float2 = (1 - 0.00000001) if float2 == 1 else float2
        float3 = (1 - 0.00000001) if float3 == 1 else float3

        divide_1 = (N//2)*2 - 1
        divide_2 = N
        
        remove_length_raw = math.floor(float1 * divide_1)
        remove_length = N//2 - remove_length_raw if N//2 - remove_length_raw > 0 else N//2 - remove_length_raw - 2
        remove_position = math.floor(float2 * divide_2)
        divide_3 = N + 1 - abs(remove_length)
        insert_position = math.floor(float3 * divide_3)
        if remove_length > 0:
            if remove_position >= insert_position:
                remove_position += 1
                if remove_position >= N:
                    remove_position = N-1

        # print(f"float1:{float1}, float2:{float2}, float3:{float3}")
        # print(f"divide_1:{divide_1}, divide_2:{divide_2}, divide_3:{divide_3}")
        # print(f"remove_length:{remove_length}, remove_position:{remove_position}, insert_position:{insert_position}")
        return remove_length, remove_position, insert_position

    def swap_and_distance(self, random_numbers:list):
        baseline_tour_copy = self.baseline_order.copy()
        remove_length, remove_position, insert_position = self.assign_sector(random_numbers[0], random_numbers[1], random_numbers[2])
        new_tour, insert_first, insert_last, insert_before, insert_after = \
            self.insert_at_position(baseline_tour_copy, remove_length, remove_position, insert_position)
        new_distance = calculate_total_distance(self.dist, new_tour)
        return new_tour, new_distance

class Search_in_same_baseline:
    def __init__(self, baseline_tour:Tour):
        self.baseline_tour = baseline_tour
        self.baseline_tour.already_baseline = True
        already_baseline_length.add(self.baseline_tour.length)
        self.baseline_tour.box_order = 0
        self.baseline_order = baseline_tour.order
        self.baseline_order_length = baseline_tour.length
        self.N = len(set(self.baseline_order))
        self.swap = Swap(self.baseline_tour, dist)
        self.search_times = 100
    
    # def plot_sobol_points(self, num_points, box_range:list):
    #     # 三次元にSobol乱数を生成
    #     sobol = Sobol(3)
    #     sobol_points = sobol.generate(num_points)
    #     # box_rangeに合うようにsobol_pointsを線形変換
    #     sobol_points[0] = sobol_points[0] * (box_range[0][1] - box_range[0][0]) + box_range[0][0]
    #     sobol_points[1] = sobol_points[1] * (box_range[1][1] - box_range[1][0]) + box_range[1][0]
    #     sobol_points[2] = sobol_points[2] * (box_range[2][1] - box_range[2][0]) + box_range[2][0]
    #     return sobol_points.T

    def plot_sobol_points(self, num_points, box_range:list):
        # 二次元平面にSobol乱数を生成
        # dnb2 = qp.DigitalNetB2(dimension=3,randomize='LMS_DS',graycode=True)
        dnb2 = qp.DigitalNetB2(dimension=3,randomize='DS',graycode=True)
        sobol_points = dnb2.gen_samples(num_points)
        # box_rangeに合うようにsobol_pointsを線形変換
        sobol_points = sobol_points.T
        sobol_points[0] = sobol_points[0] * (box_range[0][1] - box_range[0][0]) + box_range[0][0]
        sobol_points[1] = sobol_points[1] * (box_range[1][1] - box_range[1][0]) + box_range[1][0]
        sobol_points[2] = sobol_points[2] * (box_range[2][1] - box_range[2][0]) + box_range[2][0]
        return sobol_points.T
    
    def random_number_in_box(self, box_range:list):
        random1 = random.uniform(box_range[0][0], box_range[0][1])
        random2 = random.uniform(box_range[1][0], box_range[1][1])
        random3 = random.uniform(box_range[2][0], box_range[2][1])
        return [random1, random2, random3]
    
    def search(self, center_tour:Tour, use_sobol:bool=False):
        tours = [center_tour]
        box_order = center_tour.box_order
        box_range = center_tour.box()
        if use_sobol:
            sobol_array = self.plot_sobol_points(self.search_times, box_range)
        for i in range(self.search_times):
            if use_sobol:
                random_numbers = sobol_array[i]
            else:
                random_numbers = self.random_number_in_box(box_range)
            new_order, new_distance = self.swap.swap_and_distance(random_numbers)
            if new_distance == self.baseline_order_length:
                new_tour = Tour(new_distance, new_order, random_numbers, box_order+1, True)
            else:
                new_tour = Tour(new_distance, new_order, random_numbers, box_order+1)
            if (all(tour.length != new_tour.length for tour in tours) or 
            all(distance(tour.random_number, new_tour.random_number) > 0.5 for tour in tours)):
                tours.append(new_tour)

            # length 属性でソート
            tours = sorted(tours, key=lambda x: x.length)[:5]
        center_tour.box_order += 1
        if self.baseline_tour in tours:
            tours.remove(self.baseline_tour)
        return tours[:5]

    def search_all(self):
        self.baseline_tour.random_number = [0.5, 0.5, 0.5]
        tours_all = []

        new_order, new_distance = self.swap.swap_and_distance([0.5,0.0,1.0])
        new_tour = Tour(new_distance, new_order, [0.5,0.0,1.0], 0, True)
        tours_all.append(new_tour)
        new_order, new_distance = self.swap.swap_and_distance([0.5,1.0,0.0])
        new_tour = Tour(new_distance, new_order, [0.5,1.0,0.0], 0, True)
        tours_all.append(new_tour)

        tours = self.search(self.baseline_tour)
        for tour in tours:
            if (all(tour_all.length != tour.length for tour_all in tours_all) or 
                all(distance(tour.random_number, tour_all.random_number) > 0.5 for tour_all in tours)):
                    tours_all.append(tour)
        tours_all = sorted(tours_all, key=lambda x: x.length)[:5]

        for _ in range(int(math.log10(self.N))+2):
            for tour in tours_all:
                tours = self.search(tour)
                for tour in tours:
                    if (all(tour_all.length != tour.length for tour_all in tours_all) or 
                    all(distance(tour.random_number, tour_all.random_number) > 0.5 for tour_all in tours)):
                        tours_all.append(tour)
            tours_all = sorted(tours_all, key=lambda x: x.length)[:5]
        return tours_all

class Search_in_different_baseline:
    def __init__(self, first_baseline_tour:Tour):
        self.first_baseline_tour = first_baseline_tour
        # self.first_baseline_tour.already_baseline = True
        self.first_baseline_tour.box_order = 0
    
    def search(self, tours_all:list):
        # tours_all = [self.first_baseline_tour]
        tours_all_copy = tours_all.copy()
        for tour in tours_all_copy:
            tours = Search_in_same_baseline(tour).search_all()
            for tour in tours:
                if all(tour_all.length != tour.length for tour_all in tours_all):
                    tours_all_copy.append(tour)
            tours_all_copy = sorted(tours_all_copy, key=lambda x: x.length)
        return tours_all_copy[:10]
    
    def search_all(self):
        self.first_baseline_tour.already_baseline = False
        tours_all = [self.first_baseline_tour]
        start_time = time.time()
        for i in range(200):
            tours = self.search(tours_all)
            for tour in tours:
                tours_all.append(tour)
            tours_all = sorted(tours_all, key=lambda x: x.length)[:10]
            lap_time = time.time() - start_time
            print(f"i:{i:>2}, length:{tours_all[0].length:<12}, tours_all_len:{len(tours_all)},\
            num_True:{sum(tour_all.already_baseline == True for tour_all in tours_all)},\
            num_already_baseline:{sum(tour_all.length in already_baseline_length for tour_all in tours_all)}\
            lap_time:{lap_time:.5f}")
        return tours_all
        


# 適切なところで探索を打ち切り、無駄を省く
# 意味がない操作（同じところに挿入し直す）を取り除く

if __name__ == '__main__':
    # file_path = 'ATSPlib/ftv38.atsp'
    # file_path = 'ATSPlib/rbg443.atsp'
    # file_path = 'ATSPlib/ft70.atsp'
    # file_path = 'ATSPlib/rbg323.atsp'
    file_path = 'ATSPlib/rbg358.atsp'
    # file_path = 'TSPlib/eil101.tsp'
    # file_path = 'TSPlib/pr264.tsp'
    # file_path = 'TSPlib/d1655.tsp'
    # file_path = 'TSPlib/lin318.tsp'
    # file_path = 'TSPlib/p654.tsp'

    dist = read_atsp_file(file_path)
    # N = len(dist)

    # short_path = cal_shortpath(dist)
    short_path = None
    already_baseline_length = set()
    # start_time = time.time()

    baseline = Baseline_first(dist, short_path, 0)
    first_baseline_tour = Tour(baseline.distance, baseline.baseline_tour, [0.5, 0.5, 0.5], 0)

    well_done = Search_in_different_baseline(first_baseline_tour).search_all()
    print(f"最短経路は...:{well_done[0].length}")
    
    # end_time = time.time()
    # print(f"経過時間：{end_time - start_time:.1f}秒")

