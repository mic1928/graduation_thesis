from week6_3d import cal_dist, calculate_total_distance, read_tsp_file, tsp_approximation, minimum_spanning_tree, read_atsp_file
import time

from python_tsp.heuristics import solve_tsp_simulated_annealing
import numpy as np

# xopt, fopt = solve_tsp_lin_kernighan(
#     distance_matrix: np.ndarray,
#     x0: Optional[List[int]] = None,
#     log_file: Optional[str] = None,
#     verbose: bool = False,
# )



if __name__ == '__main__':
    start_time = time.time()
    # file_path = 'ATSPlib/ftv38.atsp'
    # file_path = 'ATSPlib/rbg443.atsp'
    # file_path = 'ATSPlib/ft70.atsp'
    file_path = 'ATSPlib/rbg323.atsp'
    # file_path = 'TSPlib/rbg358.atsp'
    # file_path = 'TSPlib/eil101.tsp'
    # file_path = 'TSPlib/pr264.tsp'
    # file_path = 'TSPlib/d1655.tsp'
    # file_path = 'TSPlib/lin318.tsp'
    # file_path = 'TSPlib/p654.tsp'
    
    
    if file_path[-4:] == 'atsp':
        dist = read_atsp_file(file_path)
        dist = np.array(dist)
    else:
        cities = read_tsp_file(file_path)
        dist = np.array(cal_dist(cities)) # 全てのエッジの距離が入った二次元配列
    # startpoint = 0
    # N = len(dist)

    # two_ap_tour = np.random.permutation(N)
    # two_ap_tour_length = calculate_total_distance(dist, two_ap_tour)
    # spanning_tree = minimum_spanning_tree(dist, startpoint, N) #最小全域木を生成
    # two_ap_tour = tsp_approximation(spanning_tree, startpoint)
    # two_ap_tour_length = calculate_total_distance(dist, two_ap_tour)
    # print(f"{N,len(two_ap_tour)},初期解の経路長は{two_ap_tour_length}")
    
    SA_tour, SA_tour_length = solve_tsp_simulated_annealing(dist)
    print(f"経路長は{SA_tour_length}")
    end_time = time.time()
    print(f"計算時間は{end_time-start_time:.5f}秒")