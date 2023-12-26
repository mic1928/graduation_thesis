import numpy as np
import time
import json


# class Coordinate:
#     def __init__(self, N:int):
#         self.N = N
#         array_3d = [self.integrate_process(1)]
#         repeat_time = self.N//2+1 if self.N < 256 else int(129-(129/1000)*(self.N-256))
#         repeat_time = 5 if repeat_time < 5 else repeat_time
#         # print(repeat_time)
#         for move_length in range(2, repeat_time):
#             print("あ",move_length)
#             array_2d = self.integrate_process(move_length)
#             # array_3d = np.append(array_3d, array_2d)
#             array_3d.append(array_2d)
#             array_2d = self.integrate_process(-1*move_length)
#             # array_3d = np.insert(array_3d, 0, array_2d)
#             array_3d.insert(0, array_2d)
#         self.array_3d = array_3d
#         # print(self.array_3d)

#     def create_2d_array(self,size:int):
#         n = size
#         le = self.move_length
#         array_2d = np.array([[(le, i, j) for j in range(n)] for i in range(n)])
#         return array_2d
    
#     def first_process(self, input_array):   #同じ要素を持つタプルを削除
#         co = np.copy(input_array)
#         for i in range(len(co)-1):
#             flag = 0
#             for j in range(len(co[i]) - 1, -1, -1):
#                 if co[i][j][1] == co[i][j][2]:
#                     co[i][j] = co[i+1][j]
#                     flag = 1
#                 elif flag == 1:
#                     co[i][j] = co[i+1][j]
#         return co[:-1]
    
#     def second_process(self, input_array):  #(1,0)と(0,1)は等価なので前者を削除
#         co = np.copy(input_array).tolist()
#         for i in range(len(co)):
#             for j in range(len(co[i])-1):
#                 if co[i][j][1] - co[i][j][2] == 1:
#                     co[i].remove(co[i][j])
#                     # print(f"co[i][j]:{co[i][j]},j:{j}")
#                     # print(co[i].shape)
#                     # co[i] = np.delete(co[i], j, 0)
#         return np.array(co)

#     def integrate_process(self,move_length):
#         if move_length == 1:
#             self.move_length = move_length
#             array1 = self.create_2d_array(self.N)
#             array2 = self.first_process(array1)
#             array3 = self.second_process(array2)
#             return array3
#         elif move_length > 1:
#             self.move_length = move_length
#             array1 = self.create_2d_array(self.N-move_length+2)
#             array2 = self.first_process(array1)
#             array2 = np.delete(array2, -1, 1)
#             # for row in array2:
#             #     row = np.delete(row, -1)
#             return array2
#         elif move_length < 0:
#             self.move_length = move_length
#             move_length = -1 * move_length
#             array1 = self.create_2d_array(self.N-move_length+2)
#             array2 = np.delete(array1, -1, 1)
#             # for row in array1:
#             #     row = np.delete(row, -1)
#             return array1
        
class Coordinate:
    def __init__(self, N:int):
        self.N = N
        array_3d = [self.integrate_process(1)]
        repeat_time = self.N//2+1 if self.N < 256 else int(129-(129/1000)*(self.N-256))
        repeat_time = 5 if repeat_time < 5 else repeat_time
        # repeat_time = self.N//2+1 if self.N < 256 else int(129-(129/1000)*(self.N-256))
        # repeat_time = 5 if repeat_time < 5 else repeat_time
        print(repeat_time)
        for move_length in range(2, repeat_time):
            print(move_length)
            array_2d = self.integrate_process(move_length)
            array_3d.append(array_2d)
            array_2d = self.integrate_process(-1*move_length)
            array_3d.insert(0,array_2d)
        self.array_3d = array_3d
        # print(self.array_3d)

    def create_2d_array(self,size:int):
        n = size
        le = self.move_length
        array_2d = [[(le, i, j) for j in range(n)] for i in range(n)]
        return array_2d
    
    def first_process(self, input_array):   #同じ要素を持つタプルを削除
        co = input_array.copy()
        for i in range(len(co)-1):
            flag = 0
            for j in range(len(co[i]) - 1, -1, -1):
                if co[i][j][1] == co[i][j][2]:
                    co[i][j] = co[i+1][j]
                    flag = 1
                elif flag == 1:
                    co[i][j] = co[i+1][j]
        return co[:-1]
    
    def second_process(self, input_array):  #(1,0)と(0,1)は等価なので前者を削除
        co = input_array.copy()
        for i in range(len(co)):
            for j in range(len(co[i])-1):
                if co[i][j][1] - co[i][j][2] == 1:
                    co[i].remove(co[i][j])
        return co

    def integrate_process(self,move_length):
        if move_length == 1:
            self.move_length = move_length
            array1 = self.create_2d_array(self.N)
            array2 = self.first_process(array1)
            array3 = self.second_process(array2)
            return array3
        elif move_length > 1:
            self.move_length = move_length
            array1 = self.create_2d_array(self.N-move_length+2)
            array2 = self.first_process(array1)
            for row in array2:
                row.pop()
            return array2
        elif move_length < 0:
            self.move_length = move_length
            move_length = -1 * move_length
            array1 = self.create_2d_array(self.N-move_length+2)
            for row in array1:
                row.pop()
            return array1

if __name__ == '__main__':
    # file_num = 0
    # cities = read_input(f'input/input_{file_num}.csv')
    # dist = cal_dist(cities) # 全てのエッジの距離が入った二次元配列
    # short_path = cal_shortpath(dist)
    # print("あああ")
    for i in [64,128,256,512,1024,2048,4096,8192]:
        start_time = time.time()
        array_3d = Coordinate(i).array_3d
        # JSONファイルに書き込む
        with open(f'array_3d/3d_{i}.json', 'w+') as json_file:
            json.dump(array_3d, json_file, indent=2)
        end_time = time.time()
        print(f"都市数{i}の計算時間:{end_time-start_time}")