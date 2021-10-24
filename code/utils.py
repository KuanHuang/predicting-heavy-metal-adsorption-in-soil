import numpy as np
def calculateR2(x_values, y_values):
    correlation_matrix = np.corrcoef(x_values, y_values)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2
    return r_squared

def huber(true, pred, delta):
    loss1 = np.where(np.abs(true-pred) <= delta, 0.5*((true-pred)**2), delta*np.abs(true-pred) - 0.5*(delta**2))
    return np.sum(loss1)

def logcosh(true, pred):
    loss2 = np.log(np.cosh(pred-true))
    return np.sum(loss2)