import numpy as np
import pareto
import create_cells from Create_Cells
def calc_hypervolume(y, w_ref):
    """
    calculate pareto hypervolume 

    Parameters
    ----------
    y : numpy.array
        output data
    w_ref : numpy.array
        reference point for calculating hypervolume

    Returns
    -------
    hypervolume : float
        pareto hypervolume
    """
    hypervolume = 0.0e0
    pareto_front = pareto.find_pareto_only_y(y)
    v, w = create_cells(pareto_front, w_ref)
    
    if v.ndim == 1:
        hypervolume = np.prod(w - v)
    else:
        hypervolume = np.sum(np.prod(w - v, axis=1))
    return hypervolume
