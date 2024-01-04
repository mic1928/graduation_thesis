from week6_3d import cal_dist, calculate_total_distance, read_tsp_file, tsp_approximation, minimum_spanning_tree, read_atsp_file
import time

from python_tsp.heuristics import solve_tsp_lin_kernighan
# from python_tsp.heuristics import solve_tsp_simulated_annealing
import numpy as np





if __name__ == '__main__':
    start_time = time.time()
    # file_path = 'ATSPlib/ft70.atsp'
    # file_path = 'TSPlib/eil101.tsp'
    # file_path = 'TSPlib/pr264.tsp'
    file_path = 'TSPlib/d1655.tsp'
    
    if file_path[-4:] == 'atsp':
        dist = read_atsp_file(file_path)
    else:
        cities = read_tsp_file(file_path)
        dist = np.array(cal_dist(cities)) # 全てのエッジの距離が入った二次元配列
    startpoint = 0
    N = len(dist)

    two_ap_tour = list(np.random.permutation(N))
    print(len(two_ap_tour),N)
    two_ap_tour_length = calculate_total_distance(dist, two_ap_tour)
    # spanning_tree = minimum_spanning_tree(dist, startpoint, N) #最小全域木を生成
    # two_ap_tour = tsp_approximation(spanning_tree, startpoint)
    # two_ap_tour_length = calculate_total_distance(dist, two_ap_tour)
    print(f"2近似アルゴリズムの経路長は{two_ap_tour_length}")
    if two_ap_tour[-1] == two_ap_tour[0]:
        two_ap_tour = two_ap_tour[:-1]
    xopt, fopt = solve_tsp_lin_kernighan(dist,two_ap_tour,None,False)
    
    # LK_tour, LK_tour_length = tsp_LK(dist, np.array(two_ap_tour), two_ap_tour_length)
    print(f"本物の経路長:{calculate_total_distance(dist, xopt)},経路長は{fopt}")
    end_time = time.time()
    print(f"計算時間は{end_time-start_time:.5f}秒")