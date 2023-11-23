import numpy as np
import matplotlib.pyplot as plt
from cartesian_coordinate import Swap, Coordinate, Baseline_first
from week6_haruo import read_input, cal_dist

def plot_heatmap(matrix):
    # 行列をNumPy配列に変換
    data = np.array(matrix)

    # ヒートマップを描画
    plt.imshow(data, cmap='viridis', interpolation='nearest')

    # カラーバーを表示
    plt.colorbar()

    # 軸ラベルを追加
    plt.xlabel('列')
    plt.ylabel('行')

    # for i in range(len(matrix)):
    #     for j in range(len(matrix[i])):
    #         plt.text(j, i, str(matrix[i][j]), ha='center', va='center', color='black')

    # グラフを表示
    plt.show()

# 例: 2x2行列のヒートマップを描画
# matrix_example = [[20, 30,50], [40, 50,60]]
# plot_heatmap(matrix_example)



def generate_grid_coordinates(n):
    # マスの数
    num_cells = n * n
    
    # 0から1までをn分割した値を生成
    division_points = np.linspace(0, 1, n+1)
    
    # 各マスの中央に来る値を計算して配列に追加
    coordinates = []
    for i in range(n):
        for j in range(n):
            x_center = (division_points[i] + division_points[i+1]) / 2
            y_center = (division_points[j] + division_points[j+1]) / 2
            coordinates.append((x_center, y_center))
    
    # 二次元配列に変換
    result = [coordinates[i:i+n] for i in range(0, num_cells, n)]
    return result

if __name__ == '__main__':
    file_num = 3
    baseline = Baseline_first(file_num, 0)
    dist = baseline.dist # 全てのエッジの距離が入った二次元配列
    city_num = baseline.num_cities
    coco = Coordinate(city_num)
    # print(coco)
    swsw = Swap(coco.array_3d, baseline.baseline_tour, dist)
    gege = generate_grid_coordinates(city_num-1)
    for i in range(len(gege)):
        for j in range(len(gege[i])):
            _, distance = swsw.swap_and_distance(gege[i][j][0], gege[i][j][1])
            gege[i][j] = distance
    # print(len(gege))
    plot_heatmap(gege)
    # 例: nが2の場合
    # result = generate_grid_coordinates(city_num-1)
    # print(result)

