import numpy as np
def find_pareto(X, y):
    """
    find pareto set in X and pareto frontier in y

    Paremeters
    ----------
    X : numpy.array
        input data
    y : numpy.array
        output data

    Return
    ------
    pareto_front : numpy.array
        pareto frontier in y
    pareto_set : numpy.array
        pareto set in X
    """
    y_copy = np.copy(y)
    pareto_front = np.zeros((0,y.shape[1]))
    pareto_set = np.zeros((0,X.shape[1]))
    i = 0
    j = 0
    while i < y_copy.shape[0]:
        y_outi = np.delete(y_copy, i, axis  =0)
        #paretoだったら全部false
        flag = np.all(y_outi <= y_copy[i,:],axis = 1)
        if not np.any(flag):
            pareto_front = np.append(pareto_front, [y_copy[i,:]],axis = 0)
            pareto_set = np.append(pareto_set, [X[j,:]],axis = 0)
            i += 1
        else :
            y_copy = np.delete(y_copy, i, axis= 0)
        j += 1
    return pareto_front, pareto_set

def find_pareto_only_y(y):
    """
    obtain only pareto frontier in y

    Parameters
    ----------
    y : numpy.array
        output data
    
    Returns
    -------
    pareto_front : numpy.array
        pareto frontier in y
    """
    y_copy = np.copy(y)
    pareto_front = np.zeros((0,y.shape[1]))
    i = 0
    
    while i < y_copy.shape[0]:
        y_outi = np.delete(y_copy, i, axis  =0)
        #paretoだったら全部false
        flag = np.all(y_outi <= y_copy[i,:],axis = 1)
        if not np.any(flag):
            pareto_front = np.append(pareto_front, [y_copy[i,:]],axis = 0)
            i += 1
        else :
            y_copy = np.delete(y_copy, i, axis= 0)
    return pareto_front
    
def find_pareto_from_posterior(X, mean, y):
    """
    find pareto frontier in predict mean of GPR and pareto set in X

    Parameters
    ----------
    X : numpy.array
        input data
    mean : numpy.array
        predict mean of GPR
    y : numpy.array
        output data
    
    Returns
    -------
    pareto_front : numpy.array
        pareto frontier in y defined by predict mean
    pareto_set : numpy.array
        pareto set in X
    """
    mean_copy = np.copy(mean)
    pareto_front = np.zeros((0,mean.shape[1]))
    pareto_set = np.zeros((0,X.shape[1]))
    i = 0
    j = 0
    while i < mean_copy.shape[0]:
        mean_outi = np.delete(mean_copy, i, axis  =0)
        #paretoだったら全部false
        flag = np.all(mean_outi <= mean_copy[i,:],axis = 1)
        if not np.any(flag):
            pareto_front = np.append(pareto_front, [y[j,:]],axis = 0)
            pareto_set = np.append(pareto_set, [X[j,:]],axis = 0)
            i += 1
        else :
            mean_copy = np.delete(mean_copy, i, axis= 0)
        j += 1
    return pareto_front, pareto_set