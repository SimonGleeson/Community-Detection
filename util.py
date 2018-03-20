import numpy as np
from numpy import exp, log

def logoneplusexp(mat):
    return log(1 + exp(mat))

def expoverexp(mat):
    return  exp(mat)/(1 + exp(mat))
