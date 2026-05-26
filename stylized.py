import numpy as np
import pandas as pd
from scipy.special import lambertw
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns

# Apply the default theme
sns.set_theme()
def auto_corr(ts, max_lag = 200, title = '', double_conf = False, first = True, plot = True, preset_counts = None, double_count = 1):   

    auto_corr, confidence_interval = corr(ts, ts, max_lag, preset_counts=preset_counts, double_count=double_count)
    if first:
      lags = np.arange(0,max_lag)
      auto_corr =  np.append([1], auto_corr)
      confidence_interval = np.append([2/np.sqrt(ts.size)], confidence_interval)
    else:
      lags = np.arange(1, max_lag)

    if plot:
      plt.bar(lags, auto_corr, label = 'Autocorrelation')
      plt.plot(lags,confidence_interval, color = 'red', label = r'2$\sigma$ confidence interval')
      if double_conf:
          plt.plot(lags,-confidence_interval, color = 'red')
      plt.title('ACF '+title)
      plt.xlabel('Lag')
      plt.ylabel('Autocorrelation')
      plt.legend()
      plt.savefig('ACF '+title+'.pdf')
      plt.savefig('ACF '+title+'.png')
      #plt.show()
    else:
      return auto_corr

def leverage_effect(ts, max_lag = 200, title = '', plot = True):
  leverage, _ = corr(ts,np.abs(ts)**2, max_lag)
  lags = np.arange(1,max_lag)
  if plot:
    plt.plot(lags, leverage, label = 'Leverage effect')
    plt.hlines(0,0, max_lag, linestyles = 'dotted')
    plt.xlabel(r'Lag $\tau$')
    plt.ylabel(r'L($\tau$)')
    plt.title('Leverage effect '+title)
    plt.savefig('LE '+title+'.pdf')
    plt.savefig('LE '+title+'.png')
    #plt.show()
  else:
    return leverage

def corr(ts_1, ts_2, max_lag = 200, preset_counts = None, double_count = 1):
  lags = range(1, max_lag)
  if ts_1.ndim == 1:
    ts_1 = np.expand_dims(ts_1, 0)
  if ts_2.ndim == 1:
    ts_2 = np.expand_dims(ts_2, 0)
  assert np.shape(ts_1)[1] >= max_lag, 'Decrease maximum lag ts1'
  assert np.shape(ts_2)[1] >= max_lag, 'Decrease maximum lag ts2'
  assert np.shape(ts_1) == np.shape(ts_2), 'Time series should have the same shape'
  corr_2d = np.zeros((np.shape(ts_1)[0], len(lags)))
  ts_1_mean, ts_2_mean = np.mean(ts_1), np.mean(ts_2)  
  ts_1_std, ts_2_std = np.std(ts_1), np.std(ts_2)
  counts = np.zeros(np.shape(lags)[0])
  for i, (t_1, t_2) in enumerate(zip(ts_1, ts_2)):
    for j, lag in enumerate(lags):    
      a = t_1[:-lag] - ts_1_mean
      b = t_2[lag:] - ts_2_mean
      corr_2d[i][j] = np.mean(np.multiply(a,b))/(ts_1_std*ts_2_std)
  if preset_counts is None:
    counts = np.flip(np.arange(ts_1.shape[1]-max_lag+1, ts_1.shape[1]))
    confidences_single = 2/np.sqrt(counts)
    confidences = confidences_single/np.sqrt(ts_1.shape[0])
    confidences *= np.sqrt(double_count)
  else:
    confidences = 2/np.sqrt(preset_counts)
  return np.average(corr_2d, axis = 0), confidences#http://site.iugaza.edu.ps/nbarakat/files/2010/02/Analysis_of_Time_Series_An_Introduction.pdf

def benchmark(ts, max_lag):
    ACF_bench = auto_corr(np.abs(ts), max_lag, first = False, plot = False)
    leverage_bench = leverage_effect(ts, max_lag, plot = False)
    benchmarks = np.vstack([ACF_bench, leverage_bench])
    benchmark_lags  = range(1, max_lag)
    return benchmarks, benchmark_lags


def ACF_score(generated_ts, lags, benchmarks):
  generated_ACF = auto_corr(np.abs(generated_ts), max_lag=lags[-1]+1, first=False, plot = False)
  #auto_corr(np.abs(generated_ts), max_lag=lags[-1]+1, first=False, plot = True)
  ACF_score = np.sum((benchmarks[0] - generated_ACF)**2)
  return ACF_score


def ACF_nonabs_score(generated_ts, lags):
  generated_ACF = auto_corr(generated_ts, max_lag=lags[-1]+1, first=False, plot = False)
  #auto_corr(generated_ts, max_lag=lags[-1]+1, first=False, plot = True)
  ACF_score = np.sum((generated_ACF)**2)
  return ACF_score

def leverage_score(generated_ts, lags, benchmarks):
  generated_leverage = leverage_effect(generated_ts, max_lag=lags[-1]+1, plot = False)
  #leverage_effect(generated_ts, max_lag=lags[-1]+1, plot = True)
  leverage_score = np.sum((benchmarks[1] - generated_leverage)**2)
  return leverage_score


def metrics(generated_ts, real_ts, lags, benchmarks, only_EMD = False):
  EMD = wasserstein_distance(generated_ts.flatten(), real_ts.flatten())
  if only_EMD:
      return np.array([ EMD])
  ACF_abs = ACF_score(generated_ts, lags, benchmarks)
  ACF_nonabs = ACF_nonabs_score(generated_ts, lags)
  leverage = leverage_score(generated_ts, lags, benchmarks)  
  
  #QQ_plot(generated_ts.flatten(), real_ts.flatten(), title = 'QQ plot of generated vs sample data', xlabel = 'Quantiles generated', ylabel = 'Quantiles SP 500', limit = [-0.04,0.04])
  return np.array([ACF_abs, ACF_nonabs, leverage, EMD])

def QQ_plot(data_1, data_2, title, xlabel, ylabel, limit, show = False, path = './'):
    Q_range = np.linspace(0,1,200)
    Q_data_1 = np.quantile(data_1, Q_range)
    Q_data_2 = np.quantile(data_2, Q_range)
    plt.scatter(Q_data_1, Q_data_2)
    plt.plot(Q_data_1, Q_data_1, color = 'red')
    plt.xlim(limit)
    plt.ylim(limit)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Quantiles SP 500')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(path)
    plt.clf()
    
    
def plot_gen_vs_benchmark(data_gen, data_real, ACF, ACF_err, ACF_nonabs, ACF_nonabs_err, leverage, leverage_err, ts, max_lag, ylim_ACF = [0,0.35], ylim_ACF_nonabs = [-0.1,0.1], ylim_lev = [-0.11, 0.05], double_count = 1, path = '', colours = None):
    
    # Create color palette with 4 colors from cool colormap
    if  colours is not None:
      pal = plt.cm.cool(np.linspace(0, 1, 4))
      color_real = pal[3]    # First color for real data
      color_gen = pal[1]     # Second color for generated data
      color_conf = 'red'     # Keep confidence intervals red for visibility
      error_color = pal[2]   # Third color for error visualization

    ACF_bench, CI = corr(np.abs(ts), np.abs(ts), max_lag, double_count = double_count)
    ACF_nonabs_bench = auto_corr(ts, max_lag, first = False, plot = False)
    leverage_bench = leverage_effect(ts, max_lag, plot = False)
    lags = np.arange(1,max_lag)

    ACF_gen, CI_gen = corr(np.abs(data_gen), np.abs(data_gen), max_lag, double_count = double_count)
    ACF_nonabs_gen = auto_corr(data_gen, max_lag, first = False, plot = False)
    leverage_gen = leverage_effect(data_gen, max_lag, plot = False)
    
    Q_range = np.linspace(0,1,200)
    Q_gen = np.quantile(data_gen, Q_range)
    Q_real = np.quantile(data_real, Q_range)
    
    fig, axs = plt.subplots(ncols=2, nrows=4, figsize=(10, 13)) 
    
    # Histograms with cool colors
    if colours is not None:
      axs[0,0].hist(data_real.flatten(), bins = 200, alpha = 0.4, label = 'S&P 500', density = True, color=color_real)
      axs[0,0].hist(data_gen.flatten(), bins = 100, alpha = 0.6, label = 'Generated', density = True, color=color_gen)
    else:
      axs[0,0].hist(data_real.flatten(), bins = 200, alpha = 0.4, label = 'S&P 500', density = True)
      axs[0,0].hist(data_gen.flatten(), bins = 100, alpha = 0.6, label = 'Generated', density = True)
    axs[0,0].legend()
    axs[0,0].set_xlim([-0.04,0.04])
    axs[0,0].set_title('PDF of generated log returns')
    axs[0,0].set_ylabel('PDF')
    axs[0,0].set_xlabel('Log returns')
    
    # QQ plot with cool colors
    if colours is not None:
      axs[0,1].scatter(Q_gen, Q_real, color=color_gen)
      axs[0,1].set_aspect('equal', adjustable='box')
      axs[0,1].plot(Q_real, Q_real, color = color_conf)
    else:
      axs[0,1].scatter(Q_gen, Q_real)
      axs[0,1].set_aspect('equal', adjustable='box')
      axs[0,1].plot(Q_real, Q_real, color = 'red')
    axs[0,1].set_xlim([-0.04,0.04])
    axs[0,1].set_ylim([-0.04,0.04])
    axs[0,1].set_ylabel('Quantiles SP 500')
    axs[0,1].set_xlabel('Quantiles generated')
    axs[0,1].set_title('QQ plot generated log returns')
    
    # ACF absolute returns - real data
    if colours is not None:
      axs[1,0].bar(lags, ACF_bench, label = 'Autocorrelation', color=color_real)
      axs[1,0].plot(lags, CI, color = color_conf, label = r'2$\sigma$ confidence interval')
    else:
      axs[1,0].bar(lags, ACF_bench, label = 'Autocorrelation')
      axs[1,0].plot(lags, CI, color = 'red', label = r'2$\sigma$ confidence interval')
    
    axs[1,0].set_title('ACF for absolute S&P 500 log returns')
    axs[1,0].legend()
    axs[1,0].set_ylabel('ACF')
    axs[1,0].set_xlabel(r'Lag $\tau$')
    axs[1,0].set_ylim(ylim_ACF)
    
    # ACF absolute returns - generated data
    if colours is not None:
        axs[1,1].bar(lags, ACF_gen, label = 'Autocorrelation', yerr = ACF_err, color=color_gen, ecolor=error_color)
        axs[1,1].plot(lags, CI_gen, color = color_conf, label = r'2$\sigma$ confidence interval')
    else:
        axs[1,1].bar(lags, ACF_gen, label = 'Autocorrelation', yerr = ACF_err)
        axs[1,1].plot(lags, CI_gen, color = 'red', label = r'2$\sigma$ confidence interval')
    axs[1,1].set_title('ACF for absolute generated log returns')
    axs[1,1].set_ylabel('ACF')
    axs[1,1].set_xlabel(r'Lag $\tau$')
    axs[1,1].set_ylim(ylim_ACF)
    axs[1,1].legend()
    
    # ACF non-absolute returns - real data
    if colours is not None:
        axs[2,0].bar(lags, ACF_nonabs_bench, label = 'Autocorrelation', color=color_real)
        axs[2,0].plot(lags, CI, color = color_conf, label = r'2$\sigma$ confidence interval')
        axs[2,0].plot(lags, -CI, color = color_conf)
    else:
        axs[2,0].bar(lags, ACF_nonabs_bench, label = 'Autocorrelation')
        axs[2,0].plot(lags, CI, color = 'red', label = r'2$\sigma$ confidence interval')
        axs[2,0].plot(lags, -CI, color = 'red')
    axs[2,0].set_title('ACF for S&P 500 log returns')
    axs[2,0].legend()
    axs[2,0].set_ylabel('ACF')
    axs[2,0].set_xlabel(r'Lag $\tau$')
    axs[2,0].set_ylim(ylim_ACF_nonabs)
    
    # ACF non-absolute returns - generated data
    if colours is not None:
        axs[2,1].bar(lags, ACF_nonabs_gen, label = 'Autocorrelation', yerr = ACF_nonabs_err, color=color_gen, ecolor=error_color)
        axs[2,1].plot(lags, CI_gen, color = color_conf, label = r'2$\sigma$ confidence interval')
        axs[2,1].plot(lags, -CI_gen, color = color_conf)
    else:
        axs[2,1].bar(lags, ACF_nonabs_gen, label = 'Autocorrelation', yerr = ACF_nonabs_err)
        axs[2,1].plot(lags, CI_gen, color = 'red', label = r'2$\sigma$ confidence interval')
        axs[2,1].plot(lags, -CI_gen, color = 'red')
    axs[2,1].set_title('ACF for generated log returns')
    axs[2,1].set_ylabel('ACF')
    axs[2,1].set_xlabel(r'Lag $\tau$')
    axs[2,1].set_ylim(ylim_ACF_nonabs)
    axs[2,1].legend()
    
    # Leverage effect - real data
    if colours is not None:
      axs[3,0].plot(lags, leverage_bench, label = 'Leverage effect', color=color_real)
    else:
      axs[3,0].plot(lags, leverage_bench, label = 'Leverage effect')
    axs[3,0].set_title('Leverage effect for S&P 500 log returns')
    axs[3,0].set_ylabel(r'L($\tau$)')
    axs[3,0].set_xlabel(r'Lag $\tau$')
    axs[3,0].hlines(0, 1, max_lag-1, linestyles = 'dotted', color='gray')
    axs[3,0].set_ylim(ylim_lev)
    
    # Leverage effect - generated data
    if colours is not None:
        axs[3,1].plot(lags, leverage_gen, label = 'Leverage effect', color=color_gen)
        #axs[3,1].fill_between(lags, leverage_gen-leverage_err, leverage_gen+leverage_err, alpha = 0.3, color=error_color)
    else:
        axs[3,1].plot(lags, leverage_gen, label = 'Leverage effect')
        #axs[3,1].fill_between(lags, leverage_gen-leverage_err, leverage_gen+leverage_err, alpha = 0.3, color='red')
    axs[3,1].set_title('Leverage effect for generated log returns')
    axs[3,1].set_ylabel(r'L($\tau$)')
    axs[3,1].set_xlabel(r'Lag $\tau$')
    axs[3,1].hlines(0, 1, max_lag-1, linestyles = 'dotted', color='gray')
    axs[3,1].set_ylim(ylim_lev)
    
    plt.tight_layout()
    plt.savefig(path+'/gen_vs_bench.pdf')
    #plt.show()

  
    #make the plot with jusr     # ACF non-absolute returns - generated data and save it as a seperate image
    plt.figure(figsize=(10, 5))
    if colours is not None:
        plt.bar(lags, ACF_nonabs_gen, label='Autocorrelation', yerr=ACF_nonabs_err, color=color_gen, ecolor=error_color)
        plt.plot(lags, CI_gen, color=color_conf, label=r'2$\sigma$ confidence interval')
        plt.plot(lags, -CI_gen, color=color_conf)
    else:
        plt.bar(lags, ACF_nonabs_gen, label='Autocorrelation', yerr=ACF_nonabs_err)
        plt.plot(lags, CI_gen, color='red', label=r'2$\sigma$ confidence interval')
        plt.plot(lags, -CI_gen, color='red')
    plt.title('ACF for generated log returns')
    plt.ylabel('ACF')
    plt.xlabel(r'Lag $\tau$')
    plt.ylim(ylim_ACF_nonabs)
    plt.legend()
    plt.savefig(path+'/ACF_nonabs_gen.pdf')

    # Now the same for leverage effect
    plt.figure(figsize=(10, 5))
    if colours is not None:
        plt.plot(lags, leverage_gen, label='Leverage effect', color=color_gen)
        #plt.fill_between(lags, leverage_gen-leverage_err, leverage_gen+leverage_err, alpha=0.3, color=error_color)
    else:
        plt.plot(lags, leverage_gen, label='Leverage effect')
        #plt.fill_between(lags, leverage_gen-leverage_err, leverage_gen+leverage_err, alpha=0.3, color='red')
    plt.title('Leverage effect for generated log returns')
    plt.ylabel(r'L($\tau$)')
    plt.xlabel(r'Lag $\tau$')
    plt.hlines(0, 1, max_lag-1, linestyles='dotted', color='gray')
    plt.ylim(ylim_lev)
    plt.savefig(path+'/leverage_gen.pdf')

    # Save the absolute ACF plot separately
    plt.figure(figsize=(10, 5))
    if colours is not None:
        plt.bar(lags, ACF_gen, label='Autocorrelation', yerr=ACF_err, color=color_gen, ecolor=error_color)
        plt.plot(lags, CI_gen, color=color_conf, label=r'2$\sigma$ confidence interval')
    else:
        plt.bar(lags, ACF_gen, label='Autocorrelation', yerr=ACF_err)
        plt.plot(lags, CI_gen, color='red', label=r'2$\sigma$ confidence interval')
    plt.title('ACF for absolute generated log returns')
    plt.ylabel('ACF')
    plt.xlabel(r'Lag $\tau$')
    plt.ylim(ylim_ACF)
    plt.legend()
    plt.savefig(path+'/ACF_abs_gen.pdf')
    plt.close('all')  # Close all figures to free memory