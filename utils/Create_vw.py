import numpy as np
import pareto
def create_vw(y_train, v_ref, w_ref):
    """
    create v and w from y_train, v_ref and w_ref

    Paremeters
    ----------
    y_train : numpy.array
        ovserved output data
    v_ref : numpy.array
        reference point at left lower
    w_ref : numpy.array
        reference point at right upper
    
    Returns
    -------
    v : numpy.array
        coordinates of each cell's left lower
    w : numpy.array
        coordinates of each cell's right upper
    """
    y_train_unique = np.unique(y_train, axis = 0)
    pareto_front = pareto.find_pareto_only_y(y_train_unique)
    v_base = np.vstack((v_ref, pareto_front))
    v_base = v_base[v_base[:,-1].argsort(),:]
    #とりあえずグリッドを作成
    v_all = all_grid_cells(v_base)
    #パレートフロンティアに支配されていないvだけ残す
    # dominated_v, dominated_w = create_cells(pareto_front, w_ref)
    delete = not_dominated_cell_detect(v_all, pareto_front)
    v = np.delete(v_all, delete, axis = 0)
    w_base = np.vstack((pareto_front, w_ref))
    w_base = w_base[w_base[:,-1].argsort(),:]
    w_all = all_grid_cells(w_base)
    #wも同様
    w = np.delete(w_all, delete, axis = 0)
    
    return v, w
def all_grid_cells(points):
    points_list = []
    for i in range(points.shape[1]):
        points_list.append(np.unique(points[:,i]))
    points_mesh = wrapper_meshgrid(points_list)
    all_grid = points_mesh[0].ravel()[:,None]
    for i in range(1,points.shape[1]):
        all_grid = np.hstack((np.atleast_2d(all_grid), points_mesh[i].ravel()[:,None]))
    return all_grid
def wrapper_meshgrid(args):
    return np.meshgrid(*args)
def not_dominated_cell_detect(vec, pareto_front):
    vec_tile = np.tile(vec[:,np.newaxis,:], (1, pareto_front.shape[0],1))
    pareto_front_tile = np.tile(pareto_front[np.newaxis,:,:],(vec.shape[0] ,1,1))
    true_tile = np.all(np.any(vec_tile <pareto_front_tile, axis = 2), axis = 1)
    return np.where(true_tile == False)