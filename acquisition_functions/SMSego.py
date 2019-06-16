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

class SMSego():
    """
    This class keeps attributes and methods for calculating SMSego

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
    task_num : int
        the number of objective functions
    train_num : int
        the number of observed data
    const : float
        hyperparameter of SMSego
    current_hypervolume : float
        pareto hypervolume constructed by current pareto frontier
    MOGPI : MultiOutputGPIndep
        Gaussian Process model 
    """
    def __init__(self, x_bounds, x_train, y_train, MOGPI):
        self.x_bounds = x_bounds
        self.x_train = x_train
        self.y_train = y_train
        self.w_ref = y_train.max(axis=0) + 1.0e2
        self.task_num = y_train.shape[1]
        self.train_num = y_train.shape[0]
        self.const = 1 / norm.cdf(0.5 + 1/2**self.task_num)
        self.current_hypervolume = utils.calc_hypervolume(self.y_train, self.w_ref)
        self.MOGPI = MOGPI
    def obj(self, x):
        """
        calculating smsego of x

        Paremeters 
        ----------
        x : list
            input point

        Returns 
        -------
        smsego : float
            value of SMSego
        """
        if np.any(np.all(self.x_train == x, axis=1)):
            return 1.0e5
        else:
            mean, var = self.MOGPI.predict_one(x)
            lcb = mean - self.const * np.sqrt(var)
            new_y_train = np.append(self.y_train, [lcb], axis = 0)
            new_hypervolume = utils.calc_hypervolume(new_y_train,self.w_ref)
            smsego = self.current_hypervolume - new_hypervolume
            # print(smsego)
            return smsego
    def calc_smsego(self):
        """
        optimize SMSego

        Returns
        -------
        res : res
            result of optimization by DIRECT
        """
        #現時点での獲得点が作るパレート超体積を計算
        # print(self.x_bounds)
        # self.MOGPI = MOGPI
        # res = obj(np.array([1.0]))
        #停止条件の計算
        array_bounds = np.array(self.x_bounds)
        max_bound = np.argmax(array_bounds[:,0] - array_bounds[:,1])
        terminate_vol = (0.1 ** self.x_train.shape[1]) / (array_bounds[max_bound, 1] - array_bounds[max_bound, 0])
        res = minimize(self.obj, bounds = self.x_bounds,algmethod=1,volper = terminate_vol)
        return res
