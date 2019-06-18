# -*- coding: utf-8 -*-

from numba import jit
import numpy as np
import numpy.matlib
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy import integrate
from scipy.stats import multivariate_normal
from scipy.stats import mvn
from scipy.stats import norm
import itertools as it

# @jit
def create_cells(pf, ref, ref_inv=None):
    '''
    N個のパレートフロンティアから支配された領域の排他的なセルの配列を作る. (最小化)
 
    Parameters
    ----------
    pareto frontier : numpy array
        pareto frontiers (N \times L)
    reference point : numpy array
        point that bound the objective upper space (L)
    reference point : numpy array
        point that bound the objective lower space (L) (for convinience of calculation)
 
    Retruns
    --------
    lower : numpy array
        lower position of M cells in region truncated by pareto frontier (M \times L)
    upper : numpy array
        upper position of M cells in region truncated by pareto frontier (M \times L)
    '''
    N, L = np.shape(pf)
 
    if ref_inv is None:
        ref_inv = np.min(pf, axis=0)
 
    if N == 1:
        # 1つの場合そのまま返してよし
        return np.atleast_2d(pf), np.atleast_2d(ref)
    else:
        # refと作る超体積が最も大きいものをpivotとする
        hv = np.prod(pf - ref, axis=1)
        pivot_index = np.argmax(hv)
        pivot = pf[pivot_index]
        # print('pivot :', pivot)
 
        # pivotはそのままcellになる
        lower = np.atleast_2d(pivot)
        upper = np.atleast_2d(ref)
 
        # 2^Lの全組み合わせに対して再帰を回す
        for i in it.product(range(2), repeat=L):
            # 全て1のところにはパレートフロンティアはもう無い
            # 全て0のところはシンプルなセルになるので上で既に追加済
            iter_index = np.array(list(i))==0
            if  (np.sum(iter_index) == 0) or (np.sum(iter_index) == L):
                continue
 
            # 新しい基準点(pivot座標からiの1が立っているところだけref座標に変換)
            new_ref = pivot.copy()
            new_ref[iter_index] = ref[iter_index]
 
            # 新しいlower側の基準点(計算の都合上) (下側基準点座標からiの1が立っているところだけpivot座標に変換)
            new_ref_inv = ref_inv.copy()
            new_ref_inv[iter_index] = pivot[iter_index]
 
            # new_refより全次元で大きいPareto解は残しておく必要あり
            new_pf = pf[(pf < new_ref).all(axis=1), :]
            # new_ref_invに支配されていない点はnew_refとnew_ref_invの作る超直方体に射影する
            new_pf[new_pf < new_ref_inv] = np.matlib.repmat(new_ref_inv, new_pf.shape[0], 1)[new_pf < new_ref_inv]
 
            # 再帰
            if np.size(new_pf) > 0:
                child_lower, child_upper = create_cells(new_pf, new_ref, new_ref_inv)
 
                lower = np.r_[lower, np.atleast_2d(child_lower)]
                upper = np.r_[upper, np.atleast_2d(child_upper)]
 
    return lower, upper


def main():
    L=3
    """
    pf_list = list()
    for i in it.product(range(3), repeat=L):
        if np.sum(list(i)) == 3:
            pf_list.append(list(i))
    pf = np.array(pf_list)
    """
    pf = (-1) * np.array([[3,3,2],[2,2,5],[4,4,1],[1,5,3],[5,1,4]])

    ref = np.zeros(3)
    # # 2次元実験用
    # N = 10
    # pf = np.c_[np.arange(N), np.arange(N, 0, -1)]
    print('pf : \n', pf)
    # ref = 3 * np.ones(L)

    start = time.time()
    lower, upper = create_cells(pf, ref)
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    print('lower : \n', lower)
    print('upper : \n', upper)
    print('cell_num : ', np.shape(lower)[0])
    print('HV :', np.sum(np.prod(upper - lower, axis = 1)))
    v,w = utils.split_cell(pf, ref)
    print('naive HV:', np.sum(np.prod(w - v, axis = 1)))

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(np.shape(lower)[0]):
        ax.bar3d(lower[i, 0], lower[i, 1], lower[i, 2], upper[i, 0]-lower[i, 0], upper[i, 1]-lower[i, 1] , upper[i, 2]-lower[i, 2], alpha=0.7)

    plt.show()
if __name__ == '__main__':
    main()