import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

def rms_fit(t,f, min_t=0.2, max_t = 50):

    # Fit configuration
    min_i = np.where(t<=min_t)
    max_i = np.where(t>=max_t)

    try:
        min_i = max(min_i[0][0], 0)
    except:
        min_i = 0

    try:
        max_i = min(max_i[0][0], len(t)-1)
    except:
        max_i = len(t)-1

    y_fit = f[min_i:max_i] 
    x_fit = t[min_i:max_i]

    def fit_exponential(x, a, n):
        return a * ((x)**n)
    
    params, covariance = curve_fit(fit_exponential, x_fit, y_fit)

    a, n = params
    a_v, n_v = covariance

    # print(f'params: a={a}, n={n}')
    # print(f'covariance: a={a_v}, n={n_v}')

    return a,n

def rms_initial_slope(t,f,i_range = [1,3]):

    x_slope = t[i_range[0]:i_range[1]]
    y_slope = f[i_range[0]:i_range[1]] 
    
    slope, intercept, r, p, std_err = linregress(x_slope, y_slope)

    return slope,intercept