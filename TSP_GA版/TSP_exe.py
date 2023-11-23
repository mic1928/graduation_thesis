import numpy as np
from GaTsp_copy import *
from scipy import stats
from skopt import gp_minimize

# print(calculate_point())
# print("----------------------")
N=30.0

def bayse_gatsp(param = None):
    print(f"mutation:{param[0]},population:{int(param[1])}")
    mutation = param[0]
    population = int(param[1])
    # generation = int(param[1])
    loggers = (
        Logger_trace(level=2),
        Logger_leaders(),
        Logger_fitness(),
        Logger_population(),
        Logger_population(show_breeded=True),
        Logger_last_fitness_histgram(),
    )
    loggers = (Logger_trace(level=2), Logger_leaders())
    spc_r = Species(N=int(N), seed=30)
    mdl = Model(species=spc_r, max_population = population, breedrates = mutation)
    mdl.fit(loggers=loggers)
    Logger.plot_all(loggers=loggers)
    # print(Logger.extract_epoch)
    return calculate_point()


# def calculate_point_for_bayse(param = None):
#     bayse_gatsp(param)
#     # print("----------------------")
    

# print(bayse_gatsp((0.5,800)))

# def func(param=None):
#     ret = np.cos(param[0] + 2.34) + np.cos(param[1] - 0.78)
#     print(f"param[0]:{param[0]}, param[1]:{int(param[1])}")
#     return -ret 


def calculate_95(original_result,good_result):
    # original_result = [24.962137071035635, 25.262137071035635, 28.562137071035636, 24.162137071035634, 22.562137071035636, 24.604240545510063, 23.462137071035635, 26.462137071035635, 24.162137071035634, 29.762137071035635, 25.762137071035635, 34.36213707103563, 23.762137071035635, 23.562137071035636, 25.562137071035636, 24.88434427966319, 24.462137071035635, 23.402096228486112, 23.662137071035634, 25.762137071035635, 21.962137071035635, 23.962137071035635, 24.062137071035636, 24.562137071035636, 26.062137071035636, 24.60209622848611, 23.162137071035634, 27.962137071035635, 27.762137071035635, 22.762137071035635]
    # good_result = [23.062137071035636, 27.062137071035643, 23.262137071035635, 24.162137071035634, 25.862137071035633, 25.262137071035635, 23.062137071035636, 34.562137071035636, 31.48434427966319, 23.062137071035636, 24.584344279663192, 22.78434427966319, 26.762137071035635, 22.662137071035634, 26.562137071035636, 26.862137071035633, 22.562137071035636, 30.78434427966319, 23.562137071035636, 22.962137071035635, 25.062137071035636, 24.662137071035634, 28.062137071035636, 24.78434427966319, 23.562137071035636, 23.362137071035633, 27.462137071035635, 25.662137071035634, 23.062137071035636, 27.062137071035636]
    
    # 平均を計算
    original_mean = np.mean(original_result)
    good_mean = np.mean(good_result)

    # 標準誤差を計算
    original_std_err = stats.sem(original_result)
    good_std_err = stats.sem(good_result)

    # 95%信頼区間を計算
    original_confidence_interval = stats.t.interval(0.95, len(original_result) - 1, loc=original_mean, scale=original_std_err)
    good_confidence_interval = stats.t.interval(0.95, len(good_result) - 1, loc=good_mean, scale=good_std_err)

    print(f"オリジナル版の95%信頼区間：{original_confidence_interval}")
    print(f"ベイズ版の95%信頼区間：{good_confidence_interval}")




if __name__ == '__main__':
    x1 = np.array([0.0, 1.0])
    x2 = np.array([N*30, N*40])
    x = (x1, x2)
    # x = (0,x2)
    result = gp_minimize(bayse_gatsp, x, 
                          acq_func="EI",
                          n_calls=30,
                        #   n_initial_points = 10
                          noise=0.0,
                          model_queue_size=1,
                          verbose=True)
    print(result)

    """
    original_param = [0.6,1000]
    good_param = [1.0,853.6]

    # 100回実行して結果を配列に格納
    original_results = []
    num_iterations = 100
    for i in range(num_iterations):
        print(f"{i}番目の試行")
        result = bayse_gatsp(original_param)
        original_results.append(result)
    # 結果の平均を計算
    original_average_result = np.mean(original_results)

    # 30回実行して結果を配列に格納
    good_results = []
    for i in range(num_iterations):
        print(f"{i}番目の試行")
        result = bayse_gatsp(good_param)
        good_results.append(result)
    # 結果の平均を計算
    good_average_result = np.mean(good_results)

    # 結果の表示
    print(f"オリジナル版の30回の実行結果:{original_results}, ベイズ版の30回の実行結果:{good_results}")
    print(f"オリジナル版の結果の平均:{original_average_result}, ベイズ版の結果の平均:{good_average_result}")
    calculate_95(original_results,good_results)
    



    # print(calculate_95())
    """