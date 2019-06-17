import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils))
import pareto, hypervolume
def main():
    argv = sys.argv
    data_name = argv[1]
    with open('../benchmark_functions/'+str(data_name)+'_info.json', 'r') as f:
        #データの読み込み
        json_dict = json.load(f)
        x_bounds = json_dict[data_name]['x_bounds']
        input_dim = json_dict[data_name]['input_dim']
        output_dim = json_dict[data_name]['output_dim']
    #実験設定の読み込み
    with open('../config.json','r') as f:
        json_dict = json.load(f)
        #共通設定
        #初期点の数
        initial_num = json_dict['Common']['initial_number']
        #最大イテレーション数
        max_iter = json_dict['Common']['max_iteration']
    path = './results/'+str(data_name)+'/'
    #結果を図示したいデータのディレクトリ下のディレクトリ, ファイルを取得
    directory = os.listdir(path)
    w_ref = 1.0e5 * np.one(output_dim)   
    print(directory)
    obs_points_dic = {}
    for method in directory:
        method_file = os.listdir(path + method)
        Observed_points = np.loadtxt(path + d +'/'+f,delimiter=',')
        obs_points_dic[method] = Observed_points
        comp_array = np.vstack((parego_Observed_points, w_ref))
        w_ref = np.max(comp_array, axis=0)
    for method in obs_points_dic.keys():
        current_HV = []
        for i in range(max_iter,-1, -1):
            obs_i = Observed_points[method][:Observed_points[method].shape[0] - i,:]
            buffer = hypervolume.calc_hypervolume(obs_i, w_ref) 
            current_HV.append(buffer)
        plt.plot(np.arange(max_iter), current_HV, label=method)
    plt.title('Pareto Hypervolume')
    plt.xlabel('iteration')
    plt.legend(loc='best')
    plt.savefig('../results/'+str(data_name)+'pareto_hypervolume.pdf')
    plt.close()
    
if __name__=="__main__":
    main()