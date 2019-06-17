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

We will prepare many benchmark functions for Multi-Objective Optimization. 
If you want to add a new benchmark function, you write according to our implementation.

## Installation
---
You should install some packages before using this software.
1. GPy : Gaussian Process model framework
	-  ```Bash  conda install -c conda-forge GPy```
