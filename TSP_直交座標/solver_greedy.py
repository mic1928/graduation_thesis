#!/usr/bin/env python3

import sys
import math

# from common import print_tour, read_input


def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


def solve(dist:list, cities:list, start_point:int):
    N = len(cities)

    # dist = [[0] * N for i in range(N)]
    # for i in range(N):
    #     for j in range(i, N):
    #         dist[i][j] = dist[j][i] = distance(cities[i], cities[j])

    current_city = start_point
    unvisited_cities = set(range(0, N))
    unvisited_cities.remove(current_city)
    tour = [current_city]

    while unvisited_cities:
        next_city = min(unvisited_cities,
                        key=lambda city: dist[current_city][city])
        unvisited_cities.remove(next_city)
        tour.append(next_city)
        current_city = next_city
    return tour


# if __name__ == '__main__':
#     assert len(sys.argv) > 1
#     tour = solve(read_input(sys.argv[1]))
#     print_tour(tour)
