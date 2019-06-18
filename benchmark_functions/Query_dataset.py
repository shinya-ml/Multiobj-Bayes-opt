# coding: utf-8
import json
import time
import numpy as np
from scipy.spatial import distance
from platypus import NSGAII, Problem, Real, calculate
from platypus.indicators import Hypervolume
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import axes3d
import os 
import sys
def Ackley_and_Sphere_and_Three_hump_and_Six_hump_2(X):
    """
    input space [-5, 5]^2
    """
    x = np.array(X)
    a = 20
    b = 0.2
    c = 2 * np.pi
    f_1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / 2)) - np.exp(np.sum(np.cos(c * x)) / 2) + a + np.e
    f_2 = np.sum(x**2)
    f_3 = 2 * x[0]**2 - 1.05 * x[0]**4 + x[0]**6 / 6 + x[0] * x[1] + x[1]**2
    f_4 = (4 - 2.1 * x[0]**2 + x[0]**4 / 3) * x[0]**2 + x[0] * x[1] + (4 * x[1]**2 - 4) * x[1]**2
    return [f_1, f_2,f_3,f_4]
def Ackley_and_Sphere_2(X):
    """
    input space [-5, 5]^2
    """
    x = np.array(X)
    a = 20
    b = 0.2
    c = 2 * np.pi
    f_1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / 2)) - np.exp(np.sum(np.cos(c * x)) / 2) + a + np.e
    f_2 = np.sum(x**2)
    return [f_1, f_2]
def Ackley_and_Sphere_5(X):
    """
    input space [-5, 5]^5
    """
    x = np.array(X)
    a = 20
    b = 0.2
    c = 2 * np.pi
    f_1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / 5)) - np.exp(np.sum(np.cos(c * x)) / 5) + a + np.e
    f_2 = np.sum(x**2)
    return [f_1, f_2]
def Ackley_and_Rosenbrock_5(X):
    """
    input space [-5, 5]^5
    """
    x = np.array(X)
    a = 20
    b = 0.2
    c = 2 * np.pi
    f_1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / 5)) - np.exp(np.sum(np.cos(c * x)) / 5) + a + np.e
    f_2 = 100 * (x[4] - x[3]**2)**2 + (x[3] - 1)**2 + 100 * (x[3] - x[2]**2)**2 + (x[2] - 1)**2 + 100 * (x[2] - x[1]**2)**2 + (x[1] - 1)**2 + 100 * (x[1] - x[0]**2)**2 + (x[0] - 1)**2
    return [f_1, f_2]
def Rosenbrock_and_Sphere_5(X):
    x = np.array(X)
    f_1 = 100 * (x[4] - x[3]**2)**2 + (x[3] - 1)**2 + 100 * (x[3] - x[2]**2)**2 + (x[2] - 1)**2 + 100 * (x[2] - x[1]**2)**2 + (x[1] - 1)**2 + 100 * (x[1] - x[0]**2)**2 + (x[0] - 1)**2
    f_2 = np.sum(x**2)
    return [f_1, f_2]
def Ackley_and_Rosenbrock_and_Sphere_5(X):
    x = np.array(X)
    a = 20
    b = 0.2
    c = 2 * np.pi
    f_1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / 5)) - np.exp(np.sum(np.cos(c * x)) / 5) + a + np.e
    f_2 = 100 * (x[4] - x[3]**2)**2 + (x[3] - 1)**2 + 100 * (x[3] - x[2]**2)**2 + (x[2] - 1)**2 + 100 * (x[2] - x[1]**2)**2 + (x[1] - 1)**2 + 100 * (x[1] - x[0]**2)**2 + (x[0] - 1)**2
    f_3 = np.sum(x**2)
    return [f_1, f_2, f_3]
def Branin_and_Currin(X):
    """
    f_1 = branin function
    f_2 = currin exp function(inverse)
    input space [0, 1]^2
    """
    x = np.array(X)
    f_1 = ((15 * x[1] - 5 - 5.1 * (15 * x[0] - 5)**2 / (4 * np.pi**2) - 6)**2 + (10 - 10 / (8 * np.pi)) * np.cos(15 * x[0] - 5) - 44.81) / 51.95
    f_2 = (np.exp(-0.5 / x[1]) - 1) * (2300 * x[0]**3 + 1900 * x[0]**2 + 2092 * x[0] + 60) / (100 * x[0]**3 + 500 * x[0]**2 + 4 * x[0] + 20)
    return [f_1, f_2]
def Park2_and_Hartmann(X):
    """
    f_1 = park2
    f_2 = hartmann
    """
    x = np.array(X)
    x[x == 0] = 1.0e-100
    f_1 = 2 * np.exp(x[0] + x[1]) / 3 - x[3] * np.sin(x[2]) + x[2]
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5], [0.05, 10, 17, 0.1], [3, 3.5, 1.7, 10], [17, 8, 0.05, 10]])
    P = 1.0e-4 * np.array([[1312, 1696, 5569, 124], [2329, 4135, 8307, 3736], [2348, 1451, 3522, 2883], [4047, 8828, 8732, 5743]])
    x_P = np.sum(A * (x - P)**2, axis= 1)
    f_2 = (1.1 - np.sum(alpha * np.exp(-1 * x_P))) / 0.839
    return [f_1, f_2]
def Park1_and_Park2_and_Hartmann(X):
    """
    f_1 = Park1
    f_2 = park2
    f_3 = Hartmann
    input space [0, 1]^4
    """
    x = np.array(X)
    x[x == 0] = 1.0-20
    x[x == 1] = 0.9999999999999999999
    f_1 = 0.5 * x[0] * (np.sqrt(1 + x[3] * (x[1] + x[2]**2) / x[0]**2) - 1) + (x[0] + 3 * x[3]) * np.exp(1 + np.sin(x[2]))
    f_2 = 2 * np.exp(x[0] + x[1]) / 3 - x[3] * np.sin(x[2]) + x[2]
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5], [0.05, 10, 17, 0.1], [3, 3.5, 1.7, 10], [17, 8, 0.05, 10]])
    P = 1.0e-4 * np.array([[1312, 1696, 5569, 124], [2329, 4135, 8307, 3736], [2348, 1451, 3522, 2883], [4047, 8828, 8732, 5743]])
    x_P = np.sum(A * (x - P)**2, axis= 1)
    f_3 = (1.1 - np.sum(alpha * np.exp(-1 * x_P))) / 0.839
    return [f_1, f_2, f_3]
def Booth_and_Matyas(X):
    """
    f_1 = booth function
    f_2 = matyas function
    input space [-10, 10]^2
    """
    x = np.array(X)
    f_1 = (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2
    f_1 = f_1 / 1200
    f_2 = 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]
    f_2 = f_2 / 100
    return [f_1, f_2]
def Booth_and_Dixon_price(X):
    """
    f_1 = booth function
    f_2 = Dixon price function
    input space [-10, 10]^2
    """
    x = np.array(X)
    f_1 = (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2
    f_1 = f_1 / 1200
    f_2 = (x[0] - 1)**2 + 2 * (2 * x[1]**2 - x[0])**2
    f_2 = f_2 / 9.0e4
    return [f_1, f_2]
def Dixon_price_and_Matyas(X):
    """
    f_1 = Dixon price function
    f_2 = matyas function
    input space [-10, 10]^2
    """
    x = np.array(X)
    f_1 = (x[0] - 1)**2 + 2 * (2 * x[1]**2 - x[0])**2
    f_1 = f_1 / 9.0e4
    f_2 = 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]
    f_2 = f_2 / 100
    return [f_1, f_2]
def Booth_and_Matyas_and_Dixon_price(X):
    """
    f_1 = booth function
    f_2 = matyas function
    f_3 = Dixon price function
    input space [-10, 10]^2
    """
    x = np.array(X)
    f_1 = (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2
    f_1 = f_1 / 1200
    f_2 = 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]
    f_2 = f_2 / 100
    f_3 = (x[0] - 1)**2 + 2 * (2 * x[1]**2 - x[0])**2
    f_3 = f_3 / 9.0e4
    return [f_1, f_2, f_3]
def Three_hump_and_Six_hump(X):
    """
    f_1 = three hump
    f_2 = six hump
    input space [-2, 2]^2
    """
    x = np.array(X)
    f_1 = 2 * x[0]**2 - 1.05 * x[0]**4 + x[0]**6 / 6 + x[0] * x[1] + x[1]**2
    f_2 = (4 - 2.1 * x[0]**2 + x[0]**4 / 3) * x[0]**2 + x[0] * x[1] + (4 * x[1]**2 - 4) * x[1]**2
    return [f_1, f_2]
def get_observed(data_name, X):
    return eval(data_name)(X)
def get_observed_decoupled(data_name, X, l):
    obj_values = eval(data_name)(X)
    return obj_values[l]
def get_currentratio(data_name,pareto_front_list, w_ref, input_dim, output_dim,x_bounds):
    problem = Problem(input_dim, output_dim)
    problem.types[:] = [Real(x_bounds[i][0], x_bounds[i][1]) for i in range(input_dim)]
    problem.function = eval(data_name)
    algorithm = NSGAII(problem, population_size=50)
    algorithm.run(10000)
    true_pareto_front = np.zeros((len(algorithm.result), output_dim))
    for i in range(len(algorithm.result)):
        true_pareto_front[i,:] = algorithm.result[i].objectives[:]
    # print(true_pareto_front)
    true_w_ref = np.max(true_pareto_front, axis=0) + 1.0e-2
    for i in range(w_ref.shape[0]):
        if true_w_ref[i] > w_ref[i]:
            w_ref[i] = true_w_ref[i]
    true_HV = utils.calc_hypervolume(true_pareto_front, w_ref)
    
    ratio_list = []
    for pareto_front in pareto_front_list:
        hv_i = utils.calc_hypervolume(pareto_front, w_ref)
        ratio_list.append(hv_i/true_HV)
    return ratio_list
def DTLZ1(X):
    x = np.array(X)
    g = 100 * (3 + (x[3] - 0.5)**2 + (x[4] - 0.5)**2 + (x[5] - 0.5)**2 \
    - np.cos(20 * np.pi * (x[3] - 0.5))- np.cos(20 * np.pi * (x[4] - 0.5))- np.cos(20 * np.pi * (x[5] - 0.5)))
    f_1 = (1 + g) * 0.5 * x[0] * x[1] 
    f_2 = (1 + g) * 0.5 * x[0] * x[1] * (1 - x[3]) 
    f_3 = (1 + g) * 0.5 * x[0] * (1 - x[1]) 
    f_4 = (1 + g) * 0.5 * (1 - x[0])
    return [f_1, f_2, f_3, f_4]
def DTLZ2(X):
    x = np.array(X)
    g = np.sum((x[3:] - 0.5)**2)
    f_1 = (1 + g) * np.prod(np.cos(x[:3] * 0.5 * np.pi))
    f_2 = (1 + g) * np.prod(np.cos(x[:2] * 0.5 * np.pi))
    f_3 = (1 + g) * np.cos(0.5 + np.pi * x[0]) * np.sin(0.5 * np.pi * x[1])
    f_4 = (1 + g) * np.sin(0.5 * np.pi * x[0])
    return [f_1, f_2, f_3, f_4]
def DTLZ3(X):
    x = np.array(X)
    g = 100 * (3 + (x[3] - 0.5)**2 + (x[4] - 0.5)**2 + (x[5] - 0.5)**2 \
    - np.cos(20 * np.pi * (x[3] - 0.5))- np.cos(20 * np.pi * (x[4] - 0.5))- np.cos(20 * np.pi * (x[5] - 0.5)))
    f_1 = (1 + g) * np.prod(np.cos(x[:3] * 0.5 * np.pi))
    f_2 = (1 + g) * np.prod(np.cos(x[:2] * 0.5 * np.pi))
    f_3 = (1 + g) * np.cos(0.5 * np.pi * x[0]) * np.sin(0.5 * np.pi * x[1])
    f_4 = (1 + g) * np.sin(0.5 * np.pi * x[0])
    return [f_1, f_2, f_3, f_4]
def DTLZ4(X):
    x = np.array(X)
    g = np.sum((x[3:] - 0.5)**2)
    f_1 = (1 + g) * np.prod(np.cos((x[:3]**2) * 0.5 * np.pi))
    f_2 = (1 + g) * np.prod(np.cos((x[:2]**2) * 0.5 * np.pi))
    f_3 = (1 + g) * np.cos(0.5 + np.pi * (x[0]**2)) * np.sin(0.5 * np.pi * (x[1]**2))
    f_4 = (1 + g) * np.sin(0.5 * np.pi * (x[0]**2))
    return [f_1, f_2, f_3, f_4]
def DTLZ5(X):
    x = np.array(X)
    g = np.sum((x[3:] - 0.5)**2)
    rep_x1 = (1 + 2 * g * x[1]) / (2 + 2*g)
    rep_x2 = (1 + 2 * g * x[2]) / (2 + 2*g)
    f_1 = (1 + g) * np.cos(x[0] * 0.5 * np.pi) * np.cos(rep_x1 * 0.5 * np.pi) * np.cos(rep_x2 * 0.5 * np.pi)
    f_2 = (1 + g) * np.cos(x[0] * 0.5 * np.pi) * np.cos(rep_x1 * 0.5 * np.pi)
    f_3 = (1 + g) * np.cos(0.5 + np.pi * x[0]) * np.sin(0.5 * np.pi * rep_x1)
    f_4 = (1 + g) * np.sin(0.5 * np.pi * x[0])
    return [f_1, f_2, f_3, f_4]
def DTLZ6(X):
    x = np.array(X)
    g = np.sum(np.power(x[3:], 0.1))
    rep_x1 = (1 + 2 * g * x[1]) / (2 + 2*g)
    rep_x2 = (1 + 2 * g * x[2]) / (2 + 2*g)
    f_1 = (1 + g) * np.cos(x[0] * 0.5 * np.pi) * np.cos(rep_x1 * 0.5 * np.pi) * np.cos(rep_x2 * 0.5 * np.pi)
    f_2 = (1 + g) * np.cos(x[0] * 0.5 * np.pi) * np.cos(rep_x1 * 0.5 * np.pi)
    f_3 = (1 + g) * np.cos(0.5 + np.pi * x[0]) * np.sin(0.5 * np.pi * rep_x1)
    f_4 = (1 + g) * np.sin(0.5 * np.pi * x[0])
    return [f_1, f_2, f_3, f_4]
def DTLZ7(X):
    x = np.array(X)
    g = 1 + 3 * np.sum(x[3:])
    f_1 = x[0]
    f_2 = x[1]
    f_3 = x[2]
    f_4 = (1 + g) * (4 - np.sum((x[:3] * (1 + np.sin(3 * np.pi * x[:3])))/ (1 + g)))
    return [f_1, f_2, f_3, f_4]
def DTLZ1_3d(X):
    x = np.array(X)
    g = 100 * (3+ (x[2] - 0.5)**2 + (x[3] - 0.5)**2 + (x[4] - 0.5)**2 + (x[5] - 0.5)**2 \
    - np.cos(20 * np.pi * (x[2] - 0.5))- np.cos(20 * np.pi * (x[3] - 0.5))- np.cos(20 * np.pi * (x[4] - 0.5))- np.cos(20 * np.pi * (x[5] - 0.5)))
    f_1 = (1 + g) * 0.5 * x[0] * x[1] 
    f_2 = (1 + g) * 0.5 * x[0] * (1 - x[1]) 
    f_3 = (1 + g) * 0.5 * (1 - x[0])
    return [f_1, f_2, f_3]
def DTLZ2_3d(X):
    x = np.array(X)
    g = np.sum((x[2:] - 0.5)**2)
    f_1 = (1 + g) * np.prod(np.cos(x[:2] * 0.5 * np.pi))
    f_2 = (1 + g) * np.cos(0.5 + np.pi * x[0]) * np.sin(0.5 * np.pi * x[1])
    f_3 = (1 + g) * np.sin(0.5 * np.pi * x[0])
    return [f_1, f_2, f_3]
def DTLZ3_3d(X):
    x = np.array(X)
    g = 100 * (3 + (x[2] - 0.5)**2 + (x[3] - 0.5)**2 + (x[4] - 0.5)**2 + (x[5] - 0.5)**2 \
    - np.cos(20 * np.pi * (x[3] - 0.5))- np.cos(20 * np.pi * (x[4] - 0.5))- np.cos(20 * np.pi * (x[5] - 0.5)))
    f_1 = (1 + g) * np.prod(np.cos(x[:2] * 0.5 * np.pi))
    f_2 = (1 + g) * np.cos(0.5 + np.pi * x[0]) * np.sin(0.5 * np.pi * x[1])
    f_3 = (1 + g) * np.sin(0.5 * np.pi * x[0])
    return [f_1, f_2, f_3]
def DTLZ4_3d(X):
    x = np.array(X)
    g = np.sum((x[2:] - 0.5)**2)
    f_1 = (1 + g) * np.prod(np.cos(x[:2]**2 * 0.5 * np.pi))
    f_2 = (1 + g) * np.cos(0.5 + np.pi * x[0]**2) * np.sin(0.5 * np.pi * x[1]**2)
    f_3 = (1 + g) * np.sin(0.5 * np.pi * x[0]**2)
    return [f_1, f_2, f_3]
def DTLZ5_3d(X):
    x = np.array(X)
    g = np.sum((x[3:] - 0.5)**2)
    rep_x1 = (1 + 2 * g * x[1]) / (2 + 2*g)
    f_1 = (1 + g) * np.prod(np.cos(x[:2] * 0.5 * np.pi))
    f_2 = (1 + g) * np.cos(0.5 + np.pi * x[0]) * np.sin(0.5 * np.pi * rep_x1)
    f_3 = (1 + g) * np.sin(0.5 * np.pi * x[0])
    return [f_1, f_2, f_3]
def ZDT1(X):
    x = np.array(X)
    buf = x[1:] / x[1:].shape[0]
    g = 1 + 9 * np.sum(buf)
    f_1 = X[0]
    h = 1 - np.sqrt(f_1 / g)
    f_2 = g * h
    return [f_1, f_2]
def ZDT2(X):
    x = np.array(X)
    buf = x[1:] / x[1:].shape[0]
    g = 1 + 9 * np.sum(buf)
    f_1 = X[0]
    h = 1 - np.power(f_1 / g, 2)
    f_2 = g * h
    return [f_1, f_2]
def ZDT3(X):
    x = np.array(X)
    buf = x[1:] / x[1:].shape[0]
    g = 1 + 9 * np.sum(buf)
    f_1 = X[0]
    h = 1 - np.sqrt(f_1 / g) - (f_1 / g) * np.sin(10 * np.pi * f_1)
    f_2 = g * h
    return [f_1, f_2]
def ZDT4(X):
    x = np.array(X)
    buf = x[1:] / x[1:].shape[0]
    g = 1 + 30 + np.sum(x[1:]**2 - 10 * np.cos(4 * np.pi * x[1:]))
    f_1 = X[0]
    h = 1 - np.sqrt(f_1 / g)
    f_2 = g * h
    return [f_1, f_2]
def ZDT6(X):
    x = np.array(X)
    buf = x[1:] / x[1:].shape[0]
    g = 1 + 9 * np.power(np.sum(buf), 0.25)
    f_1 = 1 - np.exp(-4 * x[0]) * np.power(np.sin(6 * np.pi * x[0]), 6)
    h = 1 - np.power(f_1 / g, 2)
    f_2 = g * h
    return [f_1, f_2]
def create_json(data_name, data_name_str, input_dim, output_dim,x_bounds):
    """
    bench_mark functionのjson
    "function_name":{
        "x_bounds":[],
        "input_dim":,
        "output_dim":
    }
    """
    
    data = {}
    data[data_name_str] = {"x_bounds":x_bounds, "input_dim":input_dim,\
    "output_dim":output_dim}
    path_file = data_name_str+'_info.json'
    with open(path_file, 'w') as f:
        json.dump(data, f)
def NSGA_test(data_name, input_dim, output_dim,x_bounds, epoch, population):
    """
    bench_mark functionのjson
    "function_name":{
        "hypervolume":,
        "v_ref":[],
        "w_ref":[],
        "x_bounds":[],
        "input_dim":,
        "output_dim":
    }
    """
    
    problem = Problem(input_dim, output_dim)
    problem.types[:] = [Real(x_bounds[i][0], x_bounds[i][1]) for i in range(input_dim)]
    problem.function = eval(data_name)
    algorithm = NSGAII(problem, population_size=population)
    start = time.perf_counter()
    algorithm.run(epoch)
    end = time.perf_counter() - start
    v_ref = []
    w_ref = []
    w_ref_norm = []
    print(len(algorithm.result))
    pareto_frontier = np.zeros((len(algorithm.result), output_dim))
    pareto_frontier_norm = np.zeros((len(algorithm.result), output_dim))
    for i in range(output_dim):
        frontier_i = np.array([s.objectives[i] for s in algorithm.result])
        pareto_frontier[:,i] = frontier_i
        min_i = np.min(frontier_i)
        max_i = np.max(frontier_i)
        frontier_i_norm = (frontier_i - min_i)/(max_i - min_i)
        pareto_frontier_norm[:,i] = frontier_i_norm
        v_ref.append(min_i)
        w_ref.append(max_i)
        w_ref_norm.append(1)

    hyp = Hypervolume(minimum=v_ref, maximum=w_ref)
    HV = hyp(algorithm.result)
    return HV, end
def calc_median(input_dim, x_bounds):
    bounds = [max(x_bounds[i]) - min(x_bounds[i]) for i in range(input_dim)]
    x = np.random.rand(50000, input_dim) * bounds + [min(x_bounds[i]) for i in range(input_dim)]
    dist_X = distance.cdist(x, x, metric="euclidean")
    median = np.median(dist_X)
    return median

def main():
    create_json(DTLZ1, "DTLZ1",6, 4, [[0, 1] for i in range(6)])
    create_json(DTLZ2, "DTLZ2",6, 4, [[0, 1] for i in range(6)])
    create_json(DTLZ3, "DTLZ3",6, 4, [[0, 1] for i in range(6)])
    create_json(DTLZ4, "DTLZ4",6, 4, [[0, 1] for i in range(6)])
    create_json(DTLZ5, "DTLZ5",6, 4, [[0, 1] for i in range(6)])
    create_json(DTLZ6, "DTLZ6",6, 4, [[0, 1] for i in range(6)])
    create_json(DTLZ7, "DTLZ7",6, 4, [[0, 1] for i in range(6)])
    create_json(ZDT1, "ZDT1",4, 2, [[0, 1] for i in range(4)])
    create_json(ZDT2, "ZDT2",4, 2, [[0, 1] for i in range(4)])
    create_json(ZDT3, "ZDT3",4, 2, [[0, 1] for i in range(4)])
    create_json(ZDT4, "ZDT4",4, 2, [[0, 1]  , [-5, 5], [-5, 5], [-5, 5]])
    create_json(ZDT6, "ZDT6",4, 2, [[0, 1] for i in range(4)])
    # create_json(MOP1, 'MOP1', 1, 2, [[-1.0e5, 1.0e5]])
    # create_json(MOP2, 'MOP2', 5, 2, [[-4.0, 4.0] for i in range(5)])
    # create_json(MOP3, 'MOP3', 2, 2, [[(-1)*np.pi, np.pi] for i in range(2)])
    # create_json(MOP4, 'MOP4', 3, 2, [[-5.0, 5.0] for i in range(3)])
    # create_json(MOP5, 'MOP5', 2, 3, [[-30.0, 30.0] for i in range(2)])
    # create_json(MOP6, 'MOP6', 2, 2, [[0, 1.0] for i in range(2)])
    # create_json(MOP7, 'MOP7', 2, 3, [[-400.0, 400.0] for i in range(2)])
    # GP_3_2([1,1,1])
    # GP_2_2_correl([1,1])
    # create_json(Booth_and_Matyas, 'Booth_and_Matyas', 2, 2, [[-10.0, 10.0] for i in range(2)])
    # create_json(Booth_and_Dixon_price, 'Booth_and_Dixon_price', 2, 2, [[-10, 10] for i in range(2)])
    # create_json(Dixon_price_and_Matyas, 'Dixon_price_and_Matyas', 2, 2, [[-10, 10] for i in range(2)])
    # create_json(Booth_and_Matyas_and_Dixon_price, 'Booth_and_Matyas_and_Dixon_price', 2, 3, [[-10, 10] for i in range(2)])
    # create_json(Three_hump_and_Six_hump,'Three_hump_and_Six_hump', 2, 2 , [[-2, 2] for i in range(2)])
    # create_json(Branin_and_Currin, 'Branin_and_Currin', 2, 2, [[0, 1] for i in range(2)])
    # create_json(Park1_and_Park2_and_Hartmann, 'Park1_and_Park2_and_Hartmann', 4, 3, [[0, 1] for i in range(4)])
    # create_json(Park2_and_Hartmann, 'Park2_and_Hartmann', 4, 2, [[0,1] for i in range(4)])
    # create_json(Ackley_and_Sphere_2, 'Ackley_and_Sphere_2', 2, 2, [[-5, 5] for i in range(2)])
    # create_json(Ackley_and_Sphere_5, 'Ackley_and_Sphere_5', 5, 2, [[-5, 5] for i in range(5)])
    # create_json(Ackley_and_Rosenbrock_5, 'Ackley_and_Rosenbrock_5', 5, 2, [[-5, 5] for i in range(5)])
    # create_json(Rosenbrock_and_Sphere_5, 'Rosenbrock_and_Sphere_5', 5, 2, [[-5, 5] for i in range(5)])
    # create_json(Ackley_and_Rosenbrock_and_Sphere_5, 'Ackley_and_Rosenbrock_and_Sphere_5', 5, 3, [[-5, 5] for i in range(5)])
    # create_json(Ackley_and_Sphere_and_Three_hump_and_Six_hump_2, 'Ackley_and_Sphere_and_Three_hump_and_Six_hump_2', 2, 4, [[-5, 5] for i in range(2)])
    

    # GP_2_2_005_005_correl([1,1])
    pass
if __name__=='__main__':
    main()
# problem = Problem(4,2)
# input_dim = 4
# problem.types[:] = [Real(0,1) for i in range(input_dim)]
# problem.function = ZDT1

# algorithm = NSGAII(problem)
# start = time.time()
# algorithm.run(10000)
# print('NSGA-II_calcurate time:{}'.format(time.time()-start))
# for solution in algorithm.result:
#     print(solution.variables + solution.objectives[:])
# hyp = Hypervolume(minimum=[0,0],maximum=[1,1])
# print(hyp(algorithm.result))
# max_ref = np.max(np.array([s.objectives[0] for s in algorithm.result]))

