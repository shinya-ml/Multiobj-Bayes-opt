# Multiobj-Bayes-opt

## Overview
---
Multi-Objective Bayesian Opitmization framework in python.
We can use acquisition functions such as 
- ParEGO
	- > [ParEGO: a hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems](https://ieeexplore.ieee.org/document/1583627)
- SMSego
	- > [Multiobjective Optimization on a Limited Budget of Evaluations Using Model-Assisted S-metric Selection](https://link.springer.com/chapter/10.1007/978-3-540-87700-4_78)
- IEIPV
	- > [The computation of the expected improvement in dominated hypervolume of Pareto front approximations](natcomp.liacs.leidenuniv.nl/material/TR-ExI.pdf)

We prepared many benchmark functions for Multi-Objective Optimization such as
1. ZDT1, 2, 3, 4, 6 (input : 4dim, output : 2dim)
2. DTLZ1 ~ 7 (input : 6dim, output : 4dim)
These information are placed as json format in benchmark_functions.
If you want to add a new benchmark function, you write according to our implementation.
If json file about benchmark function you want to experiment does not exist, experiment will　fail.
## Installation
---
You should install some packages before using this software.
We recommend using this software under anaconda environment.
1. GPy : Gaussian Process model framework
```sh
	$ conda install -c conda-forge GPy
```
2. scipydirect : A python wrapper to the DIRECT algorithm
```sh
	$ pip install scipydirect
```
3. platypus-opt : A framework for evolutionary computing in python with a focus on Multi-Objective evolutionary algorithms. 
```sh
	$ conda install -c conda-forge platypus-opt
```
## Usage
---
1. Running an experiment
	```sh
		$ python main.py method_name function_name
	```
	For example, method_name is ParEGO and function_name is ZDT1 etc...
	### Experimental settings  
	Default settings are as follows.
	- the number of initial points  : 1
	- max iteration : 20  
	If you want to these settings, please modify ./config.json.
	### Model selection
	In default setting, we optimize kernel's hyperparameters in GPR models at every iteration.
	We fixed the observation noise (= 0.0001) and variance parameter of RBF kernel (= 1.0).  
	If you want to change these settings, please modify ./models/MultiOutput_IndepGP.py. 
2. Visualization of experimental result
	```sh
		$ cd ./plot_results
		$ python plot_result.py function_name
	```
	After running plot_result.py, two pdf file will be generated in ./results/function_name/. 
	The name of these files are as follows. 
	- pareto_frontier.pdf
		- This figure shows that the **pareto-hypervolume** obtained by observed points at each iteration. 
		- **pareto-hypervolume** is popular performance measure of multi-objective bayesian optimization. 
	- pareto_hypervolume.pdf
		- This figure shows that the **pareto-frontier** obtained by observed points finally. 
## Example
---
We show the example of usage. acquisition function is ParEGO and benchmark function is ZDT1.
```sh
	$ python main.py ParEGO ZDT1
	$ cd ./plot_results
	$ python plot_result.py ZDT1
```
### Result
---
