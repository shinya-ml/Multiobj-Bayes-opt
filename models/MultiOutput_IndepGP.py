import numpy as np
import GPy

class MultiOutput_IndepGP():
    """
    Gaussian Process model for multi-output. Assuming independency among all objective functions.

    Attributes
    ----------
    model_list : list
        each element is GPR models for one objective function
    kernel_list : list
        each element is kernel 
    """
    def __init__(self):
        self.model_list = []
        self.kernel_list = []
    def fixed_ideal_params(self, x_train, y_train, lengthscale_list):
        """
        Fitting GP models when we fix hyperparameters of kernel functions

        Paremeters
        ----------
        x_train : numpy.array
            observed input data
        y_train : numpy.array  
            observed output data
        lengthscale_list : list
            ideal lengthscale for each objective function
        
        """
        input_dim = x_train.shape[1]
        output_dim = y_train.shape[1]
        self.model_list = []
        self.kernel_list = []
        # lengthscale_list = [0.5, 0.1] 
        for l in range(output_dim):
            kernel = GPy.kern.RBF(input_dim = input_dim)
            model = GPy.models.GPRegression(x_train, np.c_[y_train[:,l]],kernel=kernel)
            model['.*Gaussian_noise.variance'].constrain_fixed(1.0e-4)
            model['.*rbf.variance'].constrain_fixed(1.0)
            # model.optimize_restarts()
            model['.*rbf.lengthscale'].constrain_fixed(lengthscale_list[l])
            self.kernel_list.append(model.kern)
            self.model_list.append(model)
        print("model state")
        for model, i in enumerate(model_list):
            print("--------model for {}th object--------".format(i))
            print(model)
    def fit(self, x_train, y_train):
        """
        Fitting GP models 

        Paremeters
        ----------
        x_train : numpy.array
            observed input data
        y_train : numpy.array  
            observed output data
        
        """
        input_dim = x_train.shape[1]
        output_dim = y_train.shape[1]
        self.model_list = []
        self.kernel_list = []
        for l in range(output_dim):
            kernel = GPy.kern.RBF(input_dim = input_dim)
            model = GPy.models.GPRegression(x_train, np.c_[y_train[:,l]],kernel=kernel, normalizer=None)
            model['.*Gaussian_noise.variance'].constrain_fixed(1.0e-4)
            model['.*rbf.variance'].constrain_fixed(1.0)
            model.optimize_restarts()
            self.kernel_list.append(model.kern)
            self.model_list.append(model)
        print("model state")
        for model, i in enumerate(model_list):
            print("--------model for {}th object--------".format(i))
            print(model)
    def predict_one(self, X):
        """
        calculate predict mean and variance for an input point X

        Parameters
        ----------
        X : list
            an input point

        Returns
        -------
        mean : numpy.array
            predict mean of X
        var : numpy.array  
            predict variance of X
        """
        X_copy = np.array([X])
        pred_mean = []
        pred_var = []
        for model in self.model_list:
            mean, var = model.predict(X_copy)
            pred_mean.append(mean[0,0])
            pred_var.append(var[0,0])
        mean = np.array(pred_mean)
        var = np.array(pred_var)
        return mean, var
    def predict(self, X, full_cov=False):
        """
        calculate predict mean and variance for set of input points X

        Parameters
        ----------
        X : numpy.array
            a set of input points
        full_cov ; boolean
            wheter to calculate full covariance of X

        Returns
        -------
        pred_mean : numpy.array
            predict mean of X
        pred_var : numpy.array
            predict variance of X
            if full_cov = True, predict covariance X
        """
        # X_copy = np.array([X])
        pred_mean = np.zeros((X.shape[0], 0))
        if full_cov:
            pred_var = np.zeros((0, X.shape[0], X.shape[0]))
        else:
            pred_var = np.zeros((X.shape[0], 0))
        for model in self.model_list:
            mean, var = model.predict(X, full_cov=full_cov)
            pred_mean = np.append(pred_mean, mean, axis=1)
            if full_cov:
                pred_var = np.append(pred_var, [var], axis = 0)
            else:
                pred_var = np.append(pred_var, var, axis = 1)
        return pred_mean, pred_var
    #NSGAIIに渡す用(返り値がmeanだけ)
    def predict_NSGAII(self, X):
        """
        this method is used at multi-objective evolutionary optimization.
        difference with predict_one() is that this method return only predict mean.

        Parameters
        ----------
        X : list
            an input point 
        
        Returns
        -------
        pred_mean : list
            predict mean of X
        """
        X_copy = np.array([X])
        pred_mean = []
        pred_var = []
        for model in self.model_list:
            mean, var = model.predict(X_copy)
            pred_mean.append(mean[0,0])
            # pred_var.append(var)
        return pred_mean