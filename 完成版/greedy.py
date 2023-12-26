from week6_3d import cal_dist, calculate_total_distance, read_tsp_file, optimal_tour
import time

# def euclidean_distance(city1, city2):
#     """ユークリッド距離を計算する関数"""
#     return ((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)**0.5

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

# # 入力例
# # cost_matrix = [[0, 1, 2, 4], [5, 0, 6, 10], [8, 6, 0, 4], [1, 2, 3, 0]]

# # Greedy法による解の取得
# tour = greedy_tsp(cost_matrix)

# # 結果の表示
# print("Optimal Tour:", tour)

if __name__ == '__main__':
    start_time = time.time()
    file_path = 'TSPlib/pcb3038.tsp'
    cities = read_tsp_file(file_path)
    dist = cal_dist(cities) # 全てのエッジの距離が入った二次元配列
    startpoint = 0
    N = len(cities)
    greedy_tour = greedy_tsp(dist)
    print(f"経路長は{calculate_total_distance(dist, greedy_tour)}")
    end_time = time.time()
    print(f"計算時間は{end_time-start_time:.5f}秒")
