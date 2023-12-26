from week6_3d import cal_dist, calculate_total_distance, read_tsp_file, tsp_approximation, minimum_spanning_tree
import time

if __name__ == '__main__':
    start_time = time.time()
    file_path = 'TSPlib/pcb3038.tsp'
    cities = read_tsp_file(file_path)
    dist = cal_dist(cities) # 全てのエッジの距離が入った二次元配列
    startpoint = 0
    N = len(cities)
    spanning_tree = minimum_spanning_tree(dist, startpoint, N) #最小全域木を生成
    two_ap_tour = tsp_approximation(spanning_tree, startpoint)
    print(f"経路長は{calculate_total_distance(dist, two_ap_tour)}")
    end_time = time.time()
    print(f"計算時間は{end_time-start_time:.5f}秒")
    