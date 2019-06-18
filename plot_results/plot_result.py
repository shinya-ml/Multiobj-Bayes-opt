#coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import os, sys, json
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
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
    path = '../results/'+str(data_name)+'/'
    #結果を図示したいデータのディレクトリ下のディレクトリ, ファイルを取得
    directory = os.listdir(path)
    w_ref = np.zeros(output_dim)   
    print(directory)
    obs_points_dic = {}
    for method in directory:
        method_file = os.listdir(path + method)
        Observed_points = np.loadtxt(path + method +'/Observed_points.csv',delimiter=',')
        obs_points_dic[method] = Observed_points
        comp_array = np.vstack((Observed_points, w_ref))
        w_ref = np.max(comp_array, axis=0)
    w_ref += 1.0e-1
    for method in obs_points_dic.keys():
        current_HV = []
        for i in range(max_iter,-1, -1):
            obs_i = obs_points_dic[method][:obs_points_dic[method].shape[0] - i,:]
            buffer = hypervolume.calc_hypervolume(obs_i, w_ref) 
            current_HV.append(buffer)
        plt.plot(np.arange(max_iter+1), current_HV, label=method)
    plt.title('Pareto Hypervolume')
    plt.xlabel('iteration')
    plt.legend(loc='best')
    plt.grid(b=None, which='major')
    plt.savefig('../results/'+str(data_name)+'/pareto_hypervolume.pdf')
    plt.close()
    print('reference point : ',w_ref) 
    #plot observed pareto frontier at final iteration
    if output_dim == 2:
        if 'ParEGO' in obs_points_dic.keys():
            final_pfront = pareto.find_pareto_only_y(obs_points_dic['ParEGO'])
            pfront_sorted = final_pfront[final_pfront[:,0].argsort(),:]
            plt.scatter(pfront_sorted[:,0],pfront_sorted[:,1],c='r', label='ParEGO')
            plt.vlines(pfront_sorted[0,0],ymin=pfront_sorted[0,1],ymax=w_ref[1],colors='r')
            for i in range(pfront_sorted.shape[0]-1):
                plt.hlines(y=pfront_sorted[i,1],xmin=pfront_sorted[i,0],xmax=pfront_sorted[i+1,0],colors='r')
                plt.vlines(x=pfront_sorted[i+1,0],ymin=pfront_sorted[i+1,1],ymax=pfront_sorted[i,1],colors='r')
            plt.hlines(y=pfront_sorted[-1,1],xmin=pfront_sorted[-1,0],xmax=w_ref[0],colors='r')

        if 'IEIPV' in obs_points_dic.keys():
            final_pfront = pareto.find_pareto_only_y(obs_points_dic['IEIPV'])
            pfront_sorted = final_pfront[final_pfront[:,0].argsort(),:]
            plt.scatter(pfront_sorted[:,0],pfront_sorted[:,1],c='b', label='IEIPV')
            plt.vlines(pfront_sorted[0,0],ymin=pfront_sorted[0,1],ymax=w_ref[1],colors='b')
            for i in range(pfront_sorted.shape[0]-1):
                plt.hlines(y=pfront_sorted[i,1],xmin=pfront_sorted[i,0],xmax=pfront_sorted[i+1,0],colors='b')
                plt.vlines(x=pfront_sorted[i+1,0],ymin=pfront_sorted[i+1,1],ymax=pfront_sorted[i,1],colors='b')
            plt.hlines(y=pfront_sorted[-1,1],xmin=pfront_sorted[-1,0],xmax=w_ref[0],colors='b')
            
        if 'SMSego' in obs_points_dic.keys():
            final_pfront = pareto.find_pareto_only_y(obs_points_dic['SMSego'])
            pfront_sorted = final_pfront[final_pfront[:,0].argsort(),:]
            plt.scatter(pfront_sorted[:,0],pfront_sorted[:,1],c='g', label='SMSego')
            plt.vlines(pfront_sorted[0,0],ymin=pfront_sorted[0,1],ymax=w_ref[1],colors='g')
            for i in range(pfront_sorted.shape[0]-1):
                plt.hlines(y=pfront_sorted[i,1],xmin=pfront_sorted[i,0],xmax=pfront_sorted[i+1,0],colors='g')
                plt.vlines(x=pfront_sorted[i+1,0],ymin=pfront_sorted[i+1,1],ymax=pfront_sorted[i,1],colors='g')
            plt.hlines(y=pfront_sorted[-1,1],xmin=pfront_sorted[-1,0],xmax=w_ref[0],colors='g')
        plt.xlabel('y1')
        plt.ylabel('y2')
        plt.title('observed pareto-frontier at '+str(max_iter)+'iteration')
        plt.legend(loc='upper right')
        plt.grid(b=None, which='major')
        plt.savefig('../results/'+str(data_name)+'/pareto_frontier.pdf')
if __name__=="__main__":
    main()