from week6_3d import cal_dist, calculate_total_distance, read_tsp_file, optimal_tour, read_atsp_file, minimum_spanning_tree, tsp_approximation
import time

def two_opt(tour, cost_matrix):
    num_cities = len(tour)
    improvement = True

    while improvement:
        improvement = False

        for i in range(1, num_cities - 1):
            for j in range(i + 1, num_cities):
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                # print(new_tour)
                new_length = calculate_total_distance(cost_matrix, new_tour)

                if new_length < calculate_total_distance(cost_matrix, tour):
                    tour = new_tour
                    improvement = True

    return tour

if __name__ == '__main__':
    start_time = time.time()
    # file_path = 'ATSPlib/ftv38.atsp'
    # file_path = 'ATSPlib/rbg443.atsp'
    file_path = 'ATSPlib/ft70.atsp'
    # file_path = 'ATSPlib/rbg323.atsp'
    # file_path = 'ATSPlib/rbg358.atsp'
    # file_path = 'TSPlib/eil101.tsp'
    # file_path = 'TSPlib/pr264.tsp'
    # file_path = 'TSPlib/d1655.tsp'
    # file_path = 'TSPlib/lin318.tsp'
    # file_path = 'TSPlib/p654.tsp'


    if file_path[-4:] == 'atsp':
        dist = read_atsp_file(file_path)
        startpoint = 0
        N = len(dist)
        spanning_tree = minimum_spanning_tree(dist, startpoint, N) #最小全域木を生成
        two_ap_tour = tsp_approximation(spanning_tree, startpoint)
        two_ap_tour = two_opt(two_ap_tour, dist)
    else:
        cities = read_tsp_file(file_path)
        dist = cal_dist(cities) # 全てのエッジの距離が入った二次元配列
        startpoint = 0
        N = len(dist)
        two_ap_tour = optimal_tour(dist, startpoint, cities)
    
    print(f"経路長は{calculate_total_distance(dist, two_ap_tour)}")
    end_time = time.time()
    print(f"計算時間は{end_time-start_time:.5f}秒")