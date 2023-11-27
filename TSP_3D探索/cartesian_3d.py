from week6_haruo_3d import read_input, cal_dist, optimal_tour, calculate_total_distance, format_tour, distance, cal_shortpath

# from solver_greedy import solve
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from quasimc.sobol import Sobol
from tqdm import tqdm

class Baseline_first:
    def __init__(self, file_number:int, dist:list, short_path:list, first_point:int=0 ,baseline_tour:list=None):
        self.file_number = file_number
        self.first_point = first_point
        self.dist = dist
        self.short_path = short_path
        self.num_cities = len(self.dist)
        self.calculate_baseline_tour()

    def calculate_baseline_tour(self):
        res_tour = optimal_tour(self.dist, self.first_point, self.short_path)
        distance = calculate_total_distance(self.dist,res_tour)
        print(f"経路長:{distance}")
        self.baseline_tour = res_tour
        self.distance = distance

class Tour:
    def __init__(self, length:float, order:list, random_number:list, box_order:int ,already_baseline:bool=False):
        self.length = length
        self.order = order
        self.random_number = random_number
        self.already_baseline = already_baseline
        self.box_order = box_order
        # self.next_box_width = 10**(-(box_order+1))

    def box(self):
        self.box_width = 5**(-self.box_order)
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

    def insert_at_position(self, lst, remove_length, remove_position, insert_position):
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
    
    def assign_sector(self, float1, float2, float3): #省略の工夫なし
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
        
        # new_distance = calculate_total_distance(self.dist, new_tour)

        cut1 = self.dist[self.baseline_order[remove_position-1]][self.baseline_order[remove_position]]
        cut2 = self.dist[self.baseline_order[(remove_position+abs(remove_length)-1)%self.N]][self.baseline_order[(remove_position+abs(remove_length))%self.N]]
        cut3 = self.dist[insert_before][insert_after]
        connect1 = self.dist[insert_last][insert_after]
        connect2 = self.dist[insert_first][insert_before]
        connect3 = self.dist[self.baseline_order[remove_position-1]][self.baseline_order[(remove_position+abs(remove_length))%self.N]]
        new_distance = round(self.baseline_length - cut1 - cut2 - cut3 + connect1 + connect2 + connect3,5)
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
        self.search_times = 1000

    # def create_3d_array(self):
    #     return Coordinate(self.N).array_3d
    
    def plot_sobol_points(self, num_points, box_range:list):
        # 二次元平面にSobol乱数を生成
        # 引数は 次元数、 シード(任意)
        sobol = Sobol(3)
        sobol_points = sobol.generate(num_points)
        # box_rangeに合うようにsobol_pointsを線形変換
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
        # if self.baseline_tour in tours:
        #     tours.remove(self.baseline_tour)
        # print(box_range)
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
            # if not any(tour.length == new_tour.length for tour in tours) or not any(distance(tour.random_number, new_tour.random_number) > 0.5 for tour in tours):
            #     tours.append(new_tour)

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
        tours_all = sorted(tours_all, key=lambda x: x.length)

        for i in range(10):
            for tour in tours_all:
                tours = self.search(tour)
                for tour in tours:
                    if (all(tour_all.length != tour.length for tour_all in tours_all) or 
                    all(distance(tour.random_number, tour_all.random_number) > 0.5 for tour_all in tours)):
                        tours_all.append(tour)
            tours_all = sorted(tours_all, key=lambda x: x.length)[:5]
            # for tour in tours_all:
                # print(f"i:{i:>2}, length:{tour.length:.5f}, box_order:{tour.box_order}, random_number:{tour.random_number}")

        # for i in range(min(self.N,50)):
        # # for i in range(min(self.N,100)):
        #     min_order_tour = sorted(tours_all, key=lambda x: x.box_order)[0]
        #     tours = self.search(min_order_tour)
        #     for tour in tours:
        #         if (all(tour_all.length != tour.length for tour_all in tours_all) or 
        #         all(distance(tour.random_number, tour_all.random_number) > 0.1 for tour_all in tours)):
        #             tours_all.append(tour)
        #     print(f"box_order:{tours_all[0].box_order}")
        # tours_all = sorted(tours_all, key=lambda x: x.length)[:3]

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
            tours = Search_in_same_baseline(tour).search_all()
            for tour in tours:
                if all(tour_all.length != tour.length for tour_all in tours_all):
                    tours_all_copy.append(tour)
            tours_all_copy = sorted(tours_all_copy, key=lambda x: x.length)[:1]
            print(f"i:{i:>2}番目, length:{tours_all_copy[0].length}")
        # return sorted(tours_all_copy, key=lambda x: x.length)[:1]
        return tours_all_copy
    
    def search_all(self):
        self.first_baseline_tour.already_baseline = False
        tours_all = [self.first_baseline_tour]
        last_length = 0
        last_2_length = 0
        for i in range(100):
            tours = self.search(tours_all)
            for tour in tours:
                if all(tour_all.length != tour.length for tour_all in tours_all):
                    # if tour.length in already_baseline_length:
                    #     continue
                    tours_all.append(tour)
            tours_all = sorted(tours_all, key=lambda x: x.length)[:1]
            print(f"i:{i:>2}, length:{tours_all[0].length:<12}, tours_all_len:{len(tours_all)},\
            num_True:{sum(tour_all.already_baseline == True for tour_all in tours_all)},\
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
        # search_city_num = self.N//10
        search_city_num = 1
        for start in range(search_city_num):
            baseline = Baseline_first(self.file_number, self.dist, self.short_path, start)
            first_baseline_tour = Tour(baseline.distance, baseline.baseline_tour, [0.5, 0.5, 0.5], 0)
            tours = Search_in_different_baseline(first_baseline_tour).search_all()
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

# 適切なところで探索を打ち切り、無駄を省く
# 意味がない操作（同じところに挿入し直す）を取り除く

if __name__ == '__main__':
    file_num = 4
    cities = read_input(f'input/input_{file_num}.csv')
    dist = cal_dist(cities) # 全てのエッジの距離が入った二次元配列
    short_path = cal_shortpath(dist)
    # short_path = None
    # array_3d = Coordinate(len(cities)).array_3d
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

