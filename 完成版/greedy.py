from week6_3d import cal_dist, calculate_total_distance, read_tsp_file, read_atsp_file
import time

def greedy_tsp(cost_matrix):
    num_cities = len(cost_matrix)
    unvisited_cities = set(range(num_cities))
    current_city = 0  # 初期都市を設定
    tour = [current_city]  # 巡回路の初期化

    while unvisited_cities:
        unvisited_cities.remove(current_city)
        if unvisited_cities:
            nearest_city = min(unvisited_cities, key=lambda city: cost_matrix[current_city][city])
            tour.append(nearest_city)
            current_city = nearest_city

    # 最後に初期都市に戻る
    tour.append(0)

    return tour

if __name__ == '__main__':
    start_time = time.time()

    # file_path = 'ATSPlib/ftv38.atsp'
    file_path = 'ATSPlib/rbg443.atsp'
    # file_path = 'ATSPlib/ft70.atsp'
    # file_path = 'ATSPlib/rbg323.atsp'
    # file_path = 'ATSPlib/rbg358.atsp'
    # file_path = 'ATSPlib/ftv38.atsp'
    # file_path = 'TSPlib/eil101.tsp'
    # file_path = 'TSPlib/pr264.tsp'
    # file_path = 'TSPlib/d1655.tsp'
    # file_path = 'TSPlib/lin318.tsp'
    # file_path = 'TSPlib/p654.tsp'

    if file_path[-4:] == 'atsp':
        dist = read_atsp_file(file_path)
    else:
        cities = read_tsp_file(file_path)
        dist = cal_dist(cities) # 全てのエッジの距離が入った二次元配列
    startpoint = 0
    N = len(dist)
    greedy_tour = greedy_tsp(dist)
    print(f"経路長は{calculate_total_distance(dist, greedy_tour)}")
    end_time = time.time()
    print(f"計算時間は{end_time-start_time:.5f}秒")
