from week6_haruo import read_input, cal_dist, calculate_total_distance
from itertools import permutations

file_num = 2
cities = read_input(f'input/input_{file_num}.csv')
dist = cal_dist(cities) # 全てのエッジの距離が入った二次元配列

numbers = list(range(2, len(cities)+1))
pepe = []
i = 0
for permutation in permutations(numbers):
    if permutation[0] < permutation[-1]:
        pepe.append(permutation)
    i += 1
    if i > 100000:
        break
# print(pepe)

didi = []
for pe in pepe:
    tour = [1] + list(pe) + [1]
    tour_minus1 = [i - 1 for i in tour]
    distance = calculate_total_distance(dist, tour_minus1)
    # print(tour,distance)
    didi.append(distance)

import matplotlib.pyplot as plt

def plot_line_graph(data_list):
    # データの添字を作成（0から始める）
    x = list(range(len(data_list)))

    # 折れ線グラフをプロット
    plt.plot(x, data_list, marker='o')

    # グラフにタイトルと軸ラベルを追加（任意）
    plt.title("Line Graph")
    plt.xlabel("Index")
    plt.ylabel("Values")

    # グラフを表示
    plt.show()

# 例として、以下のリストを折れ線グラフとして表示
data = didi

plot_line_graph(data)
