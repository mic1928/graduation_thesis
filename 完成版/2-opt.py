from week6_3d import cal_dist, calculate_total_distance, read_tsp_file, optimal_tour, read_atsp_file
import time

if __name__ == '__main__':
    start_time = time.time()
    file_path = 'TSPlib/pcb3038.tsp'
    if file_path[-4:] == 'atsp':
        dist = read_atsp_file(file_path)
    else:
        cities = read_tsp_file(file_path)
        dist = cal_dist(cities) # 全てのエッジの距離が入った二次元配列
    startpoint = 0
    N = len(cities)
    two_ap_tour = optimal_tour(dist, startpoint, None)
    print(f"経路長は{calculate_total_distance(dist, two_ap_tour)}")
    end_time = time.time()
    print(f"計算時間は{end_time-start_time:.5f}秒")