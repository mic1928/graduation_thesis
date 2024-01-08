from week6_3d import cal_dist, calculate_total_distance, read_tsp_file, tsp_approximation, minimum_spanning_tree, read_atsp_file
import time

# from python_tsp.heuristics import solve_tsp_lin_kernighan
from python_tsp.heuristics import solve_tsp_simulated_annealing
from tsp_LK import tsp_LK
# import python_tsp
import numpy as np

# xopt, fopt = solve_tsp_lin_kernighan(
#     distance_matrix: np.ndarray,
#     x0: Optional[List[int]] = None,
#     log_file: Optional[str] = None,
#     verbose: bool = False,
# )



if __name__ == '__main__':
    start_time = time.time()
    file_path = 'ATSPlib/ftv38.atsp'
    # file_path = 'TSPlib/eil101.tsp'
    # file_path = 'TSPlib/d1655.tsp'
    # file_path = 'TSPlib/pcb3038.tsp'
    
    
    if file_path[-4:] == 'atsp':
        dist = read_atsp_file(file_path)
    else:
        cities = read_tsp_file(file_path)
        dist = np.array(cal_dist(cities)) # 全てのエッジの距離が入った二次元配列
    startpoint = 0
    N = len(dist)

    two_ap_tour = np.random.permutation(N)
    two_ap_tour_length = calculate_total_distance(dist, two_ap_tour)
    # spanning_tree = minimum_spanning_tree(dist, startpoint, N) #最小全域木を生成
    # two_ap_tour = tsp_approximation(spanning_tree, startpoint)
    # two_ap_tour_length = calculate_total_distance(dist, two_ap_tour)
    print(f"{N,len(two_ap_tour)},2近似アルゴリズムの経路長は{two_ap_tour_length}")
    
    LK_tour, LK_tour_length = tsp_LK(dist, np.array(two_ap_tour), two_ap_tour_length)
    print(f"本物の経路長:{calculate_total_distance(dist, LK_tour)},経路長は{LK_tour_length}")
    end_time = time.time()
    print(f"計算時間は{end_time-start_time:.5f}秒")