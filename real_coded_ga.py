#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:30:12 2021

@author: william
"""
import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga


def sphere(X):
    return sum(X**2)

#Problem Definition
problem = structure()
problem.costfunc = sphere
problem.nvar = 5
problem.varmin = -10
problem.varmax = 10

#GA Parameters
params =structure()
params.maxit = 1000
params.npop = 100
params.beta = 1
params.pc = 1
params.gamma = 0.1
params.mu = 0.1
params.sigma = 0.1

#Run GA
out = ga.run(problem, params)

#Result
#plt.plot(out.bestcost)
plt.semilogy(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel('Iterations')
plt.ylabel('Best Cost')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()