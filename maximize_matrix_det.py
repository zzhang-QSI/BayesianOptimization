__author__ = 'Zhizhuo Zhang'
from bayes_opt import BayesianOptimization
import numpy as np
import os,sys
from glob import glob
import subprocess
import argparse
from multiprocessing import Pool
import pickle


def target(**kargs):
    #assume it is square matrix
    keylist=list(map(float,kargs.keys()))
    dim=int(max(keylist))+1

    mat2d=np.zeros((dim,dim))
    for key in kargs:
        i,j=map(int,key.split("."))
        mat2d[i,j]=kargs[key]
    
    score=np.linalg.det(mat2d) ##replace to the score function you want
    print(score)
    return  score

dim=5
param={}
for i in range(dim):
    for j in range(dim):
        param[str(i)+"."+str(j)]= (0, 1)

gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
bo = BayesianOptimization(target,param, verbose=0)
bo.maximize(init_points=20, n_iter=100, acq="ei", xi=0.1,**gp_params)
print(bo.res['max'])
