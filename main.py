# coding :utf-8
import json
import numpy as np
import traceback
import time
import pickle
from scipy.spatial import distance
from platypus import NSGAII,NSGAIII, Problem, Real, calculate
import GPy
import os
import sys
import subprocess
from models import MultiOutput_IndepGP
from benchmark_functions import Query_dataset
from acquisition_functions import ParEGO, SMSego, IEIPV
def main():
    argv = sys.argv
    method = argv[1]
    data_name = argv[2]
    try:
        f = open("./benchmark_functions/" + data_name + "_info.json", "r")
    except IOError:
        print("{} does not exist".format(data_name))
        exit()
    # データの読み込み
    json_dict = json.load(f)

    x_bounds = json_dict[data_name]["x_bounds"]
    input_dim = json_dict[data_name]["input_dim"]
    output_dim = json_dict[data_name]["output_dim"]
    f.close()

    # 実験設定の読み込み
    f = open("config.json", "r")
    json_dict = json.load(f)
    # 共通設定
    # 初期点の数
    initial_num = json_dict["Common"]["initial_number"]
    # 最大イテレーション数
    max_iter = json_dict["Common"]["max_iteration"]
    # max_iter = 2
    f.close()

    # 初期点の配置
    # np.random.seed(seed)
    bounds = [max(x_bounds[i]) - min(x_bounds[i]) for i in range(input_dim)]
    # print(bounds)
    initial_x = np.random.rand(initial_num, input_dim) * bounds + [min(x_bounds[i]) for i in range(input_dim)]
    initial_y = np.array([Query_dataset.get_observed(data_name, initial_x[i, :]) for i in range(initial_num)])
    # inference ratio
    MOGPI = MultiOutput_IndepGP.MultiOutput_IndepGP()
    MOGPI.fit(initial_x, initial_y)
    obs_y_list = [initial_y]
    start = time.time()
    for i in range(max_iter):
        print("{}th iteration".format(i))
        if method == "ParEGO":
            rho = json_dict["ParEGO"]["rho"]
            xi = json_dict["ParEGO"]["xi"]
            parego = ParEGO.ParEGO(x_bounds, initial_x, initial_y, rho, xi)
            res = parego.calc_parego()
        elif method == "SMSego":
            beta = json_dict["SMSego"]["beta"]
            smsego = SMSego.SMSego(x_bounds, initial_x, initial_y, MOGPI)
            res = smsego.calc_smsego()
        elif method == "IEIPV":
            ieipv = IEIPV.IEIPV(x_bounds, initial_x, initial_y)
            res = ieipv.calc_ieipv(MOGPI)
            # res = IEIPV.calc_ieipv(MTGPR)
        print('-----result of DIRECT-----')
        print(res)
        initial_x = np.append(initial_x, [res.x], axis=0)
        next_y = Query_dataset.get_observed(data_name, res.x)
        initial_y = np.append(initial_y, [next_y], axis=0)
        obs_y_list.append(initial_y)
        MOGPI.fit(initial_x, initial_y)
    cmd = 'mkdir -p ./results/'+str(data_name)+'/'+str(method)
    subprocess.check_call(cmd.split())
    np.savetxt("./results/"+ str(data_name)+ "/"+ str(method)+ "/Observed_points.csv",np.array(initial_y),delimiter=",")
        
    print("avg time[s]: {}".format(time.time() - start))
if __name__=="__main__":
    main()