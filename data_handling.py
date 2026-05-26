import numpy as np
import pandas as pd
from scipy.special import lambertw
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import os


def chopchop(data, window_size, stride=1):
    '''
    strides over the data with a window of size window_size and returns the resulting array
    
    '''
    return np.lib.stride_tricks.sliding_window_view(data, window_shape = window_size)[::stride]


def transform(data, delta = 0.5, unity_transform = False):#0.172
    #first normalisation
    mu1 = np.mean(data)
    s1 = np.std(data)
    data_norm1 = (data-mu1)/s1

    #Lambert inverse transformation
    data_lambert = np.real(np.sign(data_norm1)*(lambertw(delta*data_norm1**2)/delta)**(0.5))

    #second normalisation
    mu2 = np.mean(data_lambert)
    s2 = np.std(data_lambert)
    data_norm2 = (data_lambert-mu2)/s2
    
    #transformation to [0,1] range
    minimum = np.quantile(data_norm2, 0.0005)
    maximum = np.quantile(data_norm2, 0.9995)
    if unity_transform:
        data_norm2 -= minimum
        data_norm2 /= maximum-minimum
        data_norm2 = np.where(data_norm2>0, data_norm2, 0)
        data_norm2 = np.where(data_norm2<1, data_norm2, 1)
    params = [mu1, s1, mu2, s2, delta, minimum, maximum, unity_transform]
    return data_norm2, params

def inverse_transform(data, params): 
    mu1, s1, mu2, s2, delta, minimum, maximum, unity_transform = params
    if unity_transform:
        data_norm2 = data *(maximum-minimum)
        data_norm2 += minimum
    else:
        mask1 = data >= minimum
        mask2 = data <= maximum
        
        data_norm1 = np.where(mask1, data, minimum)
        data_norm2 = np.where(mask2, data_norm1, maximum)
    #invert second normalisation
    data_lambert = data_norm2*s2+mu2

    
    #invert lambert
    data_norm1 = data_lambert*np.exp(delta*data_lambert**2/2)
    #invert first normalisation
    data = data_norm1*s1+mu1
    return data

def load_SP500_data(data_time):
    if data_time == 'daily':
        return load_SP500_lr()
    elif data_time == 'weekly':
        return load_SP500_lr_week_long()
    else:
        return load_SP500_lr_own(data_time)

def load_SP500_lr():
    _sp500 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SP500')
    data_SP500 = pd.read_csv(_sp500, sep = ';')
    data_SP500 = data_SP500.replace(',','', regex=True)
    open_sp500 = data_SP500['Close*'].to_numpy()
    sp500 = open_sp500.astype('float64')
    sp500 = np.flip(sp500)
    log_returns_sp500 = np.log(sp500[1:])-np.log(sp500[:-1])
    return log_returns_sp500

def load_SP500_lr_week_long():
    data_SP500 = pd.read_csv('/Fin_Data/weekly_data.csv', sep = ';')
    data_SP500 = data_SP500.replace(',','', regex=True)
    open_sp500 = data_SP500['Close*'].to_numpy()
    sp500 = open_sp500.astype('float64')
    sp500 = np.flip(sp500)
    log_returns_sp500 = np.log(sp500[1:])-np.log(sp500[:-1])
    return log_returns_sp500

def load_SP500_lr_own(name):
    #print current directory
    print(os.getcwd())
    data_SP500 = pd.read_csv("/Fin_Data/" + name, sep = ',')
    open_sp500 = data_SP500['close'].to_numpy()
    sp500 = open_sp500.astype('float64')
    log_returns_sp500 = np.log(sp500[1:])-np.log(sp500[:-1])
    return log_returns_sp500

def mean_and_error(data, flatten = True):
    #mean and error over axis = 1
    data_average = np.average(data, axis = 1)
    data_err = np.std(data, axis = 1, ddof = 1)/np.sqrt(data.shape[1])
    if flatten:
        data_average, data_err = data_average.flatten(), data_err.flatten()
    return data_average, data_err