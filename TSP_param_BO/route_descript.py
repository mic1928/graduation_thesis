#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 23:21:46 2023

@author: tomo.f
"""

import matplotlib.pyplot as plt
# from bayes_opt import BayesianOptimization

# 都市の座標辞書
city_coordinates = {'1': (43, 93), '2': (57, 93), '3': (63, 2), '4': (96, 65), '5': (25, 4), '6': (99, 40), '7': (54, 28), '8': (60, 1), '9': (78, 65), '10': (45, 35), '11': (93, 86), '12': (24, 24), '13': (79, 53), '14': (56, 94), '15': (94, 38), '16': (41, 10), '17': (58, 70), '18': (47, 4), '19': (56, 21), '20': (61, 73), '21': (85, 72), '22': (88, 84), '23': (80, 95), '24': (81, 78), '25': (40, 31), '26': (91, 86), '27': (78, 49), '28': (49, 12), '29': (36, 36), '30': (25, 66)}

# ルート（都市の順序リスト）
route = ['6', '27', '13', '9', '4', '21', '24', '22', '26', '11', '23', '20', '17', '29', '25', '16', '28', '19', '7', '10', '2', '14', '1', '30', '12', '5', '18', '8', '3', '15']
# ルートを順序通りに座標リストに変換
route_coordinates = [city_coordinates[city] for city in route]

# 閉路を描画
x, y = zip(*route_coordinates)  # x座標とy座標を分離
plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o', linestyle='-', markersize=10)
plt.title('TSP Route')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)

# 閉路を表示
for i, city in enumerate(route):
    plt.annotate(city, (x[i], y[i]), fontsize=12, ha='center', va='bottom')

# 最初の都市から最後の都市に戻る閉路を示すために線を追加
plt.plot((x[0], x[-1]), (y[0], y[-1]), linestyle='--', color='gray')

plt.show()
