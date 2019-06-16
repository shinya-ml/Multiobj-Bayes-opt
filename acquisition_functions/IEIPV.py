import numpy as np
from scipy.stats import *
from scipy.spatial import distance
import GPy
from scipydirect import minimize
import utils
import MultiTaskGPR as MTGP
from Create_Cells import create_cells
import warnings
import time
from joblib import Parallel, delayed
from multiprocessing import Pool
import multiprocessing as multi
class IEIPV():
    """
    This class keeps attributes and methods for calculating IEIPV

    Attributes
    ----------
    x_bounds : list
        input domain which is optimized.
    x_train : numpy.array
        observed input data
    y_train : numpy.array
        observed output data
    w_ref : numpy.array
        reference point at right upper
    v_ref : numpy.array
        reference point ar left lower
    task_num : int
        the number of objective functions
    train_num : int
        the number of observed data
    """
    def __init__(self, x_bounds, x_train, y_train):
        self.x_bounds = x_bounds
        self.x_train = x_train
        self.y_train = y_train
        self.w_ref = y_train.max(axis=0) + 1.0e-2
        self.v_ref = y_train.min(axis=0) - 1.0
        self.task_num = y_train.shape[1]
        self.train_num = y_train.shape[0]
    def calc_ieipv(self, MOGPI):
        """
        obtain the results optimize IEIPV

        Parameters
        ----------
        MOGPI : MultiOutputGPIndep
            Gaussian Process Regression model for each objective function
        Returns
        -------
        res : res
            result of optimization by DIRECT
        """
        #cellの作成
        v, w = utils.create_vw(self.y_train, self.v_ref, self.w_ref)
        def obj(x):
            if np.any(np.all(self.x_train == x, axis=1)):
                return 1.0e5
            else:
                # mean, var = MTGPR.multitaskGP_predict(np.atleast_2d(x))
                mean, var = MOGPI.predict_one(x)
                alpha = (mean - v) / np.sqrt(var)
                beta = (mean - w) / np.sqrt(var)
                ieipv_each_cell = var * ((norm.pdf(beta) - norm.pdf(alpha)) + beta * (norm.cdf(beta) - norm.cdf(alpha)))
                ieipv = (-1) * np.sum(np.prod(ieipv_each_cell, axis = 1))
                return ieipv
        #停止条件の計算
        array_bounds = np.array(self.x_bounds)
        max_bound = np.argmax(array_bounds[:,0] - array_bounds[:,1])
        terminate_vol = (0.1 ** self.x_train.shape[1]) / (array_bounds[max_bound, 1] - array_bounds[max_bound, 0])
        res = minimize(obj, bounds = self.x_bounds, algmethod=1, volper = terminate_vol)
        # print(obj([1]))
        return res