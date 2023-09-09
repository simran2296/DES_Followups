# Functions used in making KN contour plots

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy import ndimage

from scipy.stats import poisson

from matplotlib import rcParams
rcParams['font.family'] = 'serif'


def get_eff_data(filename, gridsize=100):
    # load efficiency data
    df = pd.read_csv(filename)

    # interpolate over grid
    interpolator = LinearNDInterpolator(df[['VK', 'LOGXLAN', 'LOGMASS']].values, df['EFFICIENCY'].values)
    vk_range = np.linspace(np.min(df['VK'].values), np.max(df['VK'].values), gridsize)
    logmass_range = np.linspace(np.min(df['LOGMASS'].values), np.max(df['LOGMASS'].values), gridsize)
    logxlan_range = np.linspace(np.min(df['LOGXLAN'].values), np.max(df['LOGXLAN'].values), gridsize)

    grid_list = []
    for vk in vk_range:
        for logxlan in logxlan_range:
            for logmass in logmass_range:
                grid_list.append((vk, logxlan, logmass))

    grid = np.array(grid_list)
    del grid_list

    k = interpolator(grid)
    arr = np.vstack((k, grid.T))
    eff_df = pd.DataFrame(data=arr.T, columns=['EFFICIENCY', 'VK', 'LOGXLAN', 'LOGMASS'])
    eff_df = eff_df.dropna()
    return eff_df

def calc_prob(eff, B, lam, f):
    #return 1. / (lam * f * eff + B) 
    return (1. - eff) * poisson.pmf(mu=B, k=0)

def get_prob_df(df1, B, lam, f):
    probs = calc_prob(df1['EFFICIENCY'].values, B, lam, f)
    probs /= np.sum(probs)
    prob_df = df1.copy()
    prob_df['PROBABILITY'] = probs
    return prob_df

def get_samples(prob_df, size):
    indices = np.random.choice(np.arange(prob_df.shape[0]), size=size, p=prob_df['PROBABILITY'].values)
    return prob_df.iloc[indices][['VK', 'LOGXLAN', 'LOGMASS']]

def samples_with_background(df1, B, lam=1.0, f=1.0, bins=100, num_samples=1000000):
    prob_df = get_prob_df(df1, B, lam, f)
    samples = get_samples(prob_df, num_samples)
    return samples

def make_plot_data(sample_df, param_1, param_2, sigma=4, bins=100):
    #make 2d histogram
    H, xedges, yedges = np.histogram2d(sample_df[param_1], sample_df[param_2], bins=bins, normed=True)
    x_centers = [0.5 * (xedges[i] + xedges[i+1]) for i in range(len(xedges) - 1)]
    y_centers = [0.5 * (yedges[i] + yedges[i+1]) for i in range(len(yedges) - 1)]
    
    #smooth noisy data
    data = ndimage.filters.gaussian_filter(H, sigma)
    
    #make a grid
    Y, X = np.meshgrid(x_centers, y_centers)
    
    return {'X': X, 'Y': Y, 'data': data}

def make_full_plot_data(filenames, backgrounds, lam=1.0, f=1.0, sigma_list=[{'20': 4, '21': 6, '12': 7}]):
    if len(sigma_list) != len(filenames):
        sigma_list = [sigma_list[0]] * len(filenames)
    
    full_plot_data = []
    for filename, B, sigmas in zip(filenames, backgrounds, sigma_list):

        eff_df = get_eff_data(filename)
        prob_df = get_prob_df(eff_df, B, lam=lam, f=f)
        sample_df = samples_with_background(prob_df, B, lam=lam, f=f)

        plot_data = {}
        plot_data['VK_LOGMASS'] = make_plot_data(sample_df, 'VK', 'LOGMASS', sigma=sigmas['20'])
        plot_data['LOGXLAN_LOGMASS'] = make_plot_data(sample_df, 'LOGXLAN', 'LOGMASS', sigma=sigmas['21'])
        plot_data['VK_LOGXLAN'] = make_plot_data(sample_df, 'VK', 'LOGXLAN', sigma=sigmas['12'])

        full_plot_data.append(plot_data)
        
    return full_plot_data


def make_contour(info, ax):
    
    ax.contourf(info['Y'], info['X'], info['data'].T, 
                levels=np.percentile(info['data'], [0, 5, 61]),
                colors=['#0e30d8', '#b6bfec'],
                alpha=0.3)
    l = ax.contour(info['Y'], info['X'], info['data'].T, 
               levels=np.percentile(info['data'], [5, 61, 90]),
               colors=['black'],
               linestyles=['solid', 'solid', 'dashed'],
               linewidths=[2,2])
    labels = [r'$2\sigma$', r'$1\sigma$', '10%']
    fmt = {}
    for lab, s in zip(l.levels, labels):
        fmt[lab] = s
    ax.clabel(l, fmt=fmt, fontsize=14)
    
    return ax

def get_boundaries(bindata, info, levels=[5, 39.3]):
    cutoffs = np.percentile(info, q=levels)
    
    outdata = []
    for cutoff in cutoffs:
        outdata.append(get_boundaries_single(bindata, info, cutoff))
        
    return outdata
        
def get_boundaries_single(bindata, info, cutoff, thresh=0.02):
    idx = np.arange(len(bindata))
    
    num_possibilities = 3
    while num_possibilities > 2:
        thresh /= 1.9 #1.4
    
        cond = (info < cutoff + thresh) & (info > cutoff - thresh)
        num_possibilities = len(idx[cond])

    #print(idx[cond])
    helper = False
    if len(idx[cond]) == 2:
        #two sided constraints
        if np.abs(idx[cond][0] - idx[cond][1]) != 1:
            # true two sided limit
            return idx[cond]
        helper = True
    
    #one sided limit
    if helper:
        up_low = np.argmin(np.array([np.abs(np.max(bindata) - bindata[idx[cond][0]]), np.abs(np.min(bindata) - bindata[idx[cond][0]])]))
    else:
        up_low = np.argmin(np.array([np.abs(np.max(bindata) - bindata[idx[cond]]), np.abs(np.min(bindata) - bindata[idx[cond]])]))
    #print([np.abs(np.max(bindata) - bindata[idx[cond]]), np.abs(np.min(bindata) - bindata[idx[cond]])])
    if up_low == 0:
        return np.array([idx[cond][0], len(bindata) -1])
    else:
        return np.array([0, idx[cond][0]])


def make_hist(info, ax, hist_params):
    #ax.hist(info, bins=20)

    ax.step(info[0], info[1], **hist_params)


    boundary_idx = get_boundaries(info[0], info[1], levels=[5, 39.3])

    #one sigma
    ax.fill_between(info[0][boundary_idx[1][0]:boundary_idx[1][1],0],
                    np.zeros(boundary_idx[1][1] - boundary_idx[1][0]),
                    np.array(info[1])[boundary_idx[1][0]:boundary_idx[1][1]],
                    alpha=0.3, color='#b6bfec')
    
    #two sigma
    ax.fill_between(info[0][boundary_idx[0][0]:boundary_idx[0][1],0],
                    np.zeros(boundary_idx[0][1] - boundary_idx[0][0]),
                    np.array(info[1])[boundary_idx[0][0]:boundary_idx[0][1]],
                    alpha=0.3, color='#0e30d8')


    plot_range = np.max(info[1]) - np.min(info[1])

    ax.set_ylim(np.min(info[1]) - 0.05 * plot_range, np.max(info[1]) + 0.05 * plot_range)


    #Don't ask me what the heck is going on with the dimensions here
    print("One Sigma")
    print("%.5f  %.5f" %(info[0][boundary_idx[1][0]][0], info[0][boundary_idx[1][1]][0]))
    print("Two Sigma")
    print("%.5f  %.5f" %(info[0][boundary_idx[0][0]][0], info[0][boundary_idx[0][1]][0]))

    print('\n')

    return ax


def make_plot_full(all_info, plot_contour_list=[True], hist_param_list=[{}], outfile=''):
    if len(plot_contour_list) != len(all_info):
        plot_contour_list = [True] * len(all_info)
    if len(hist_param_list) != len(all_info):
        hist_param_list = [{}] * len(all_info)
    
    fig, axs = plt.subplots(3, 3, figsize=(12,12), dpi=120)
    #fig.tight_layout()
    
    #PLOT ALL DATA
    for full_info, plot_contours, hist_params in zip(all_info, plot_contour_list, hist_param_list):
        axs = make_plot_single(full_info, axs, plot_contours=plot_contours, hist_params=hist_params)

    
    ## Make it pretty
    
    #VK vs LOGMASS
    ax = axs[2,0]
    ax.set_xlabel('$v_{ej}$ [$c$]', fontsize=14)
    ax.set_ylabel("$M_{ej}$ [$M_\odot$]", fontsize=16)
    ax.set_xticklabels(['%.2f' %x for x in ax.get_xticks()], fontsize=12)
    ax.set_yticklabels(['%.3f' %(10**x) for x in ax.get_yticks()], fontsize=12)
    ax.set_xlim(0.03, 0.3)
    ax.set_ylim(-3,-1)
    ax.text(0.15, -1.8, 'Excluded at $1\sigma$\nConfidence Level', fontsize=14,
            horizontalalignment='center', verticalalignment='center')
    
    #VK vs LOGXLAN
    ax = axs[1,0]
    ax.set_ylabel('$\log (X_{lan})$', fontsize=14)
    ax.set_xlim(0.03, 0.3)
    ax.set_ylim(-9, -1)
    ax.set_xticklabels([''])
    ax.set_yticklabels(['%i' %x for x in ax.get_yticks()], fontsize=12)
    ax.text(0.15, -6.5, 'Excluded at $1\sigma$\nConfidence Level', fontsize=14,
            horizontalalignment='center', verticalalignment='center')
    
    #LOGXLAN vs LOGMASS
    ax = axs[2,1]
    ax.set_xlabel('$\log (X_{lan})$', fontsize=14)
    ax.set_xlim(-9, -1)
    ax.set_ylim(-3,-1)
    ax.set_xticklabels(['%i' %x for x in ax.get_xticks()], fontsize=12)
    ax.set_yticklabels('')
    ax.text(-5, -1.8, 'Excluded at $1\sigma$\nConfidence Level', fontsize=14,
            horizontalalignment='center', verticalalignment='center')
    
    #VK
    ax = axs[0,0]
    ax.set_yticklabels('')
    ax.tick_params(direction='in')
    ax.set_xlim(0.03, 0.3)
    ax.set_xticklabels([''])
    
    #LOGXLAN
    ax = axs[1,1]
    ax.set_yticklabels('')
    ax.tick_params(direction='in')
    ax.set_xlim(-9, -1)
    ax.set_xticklabels([''])
    
    #LOGMASS
    ax = axs[2,2]
    ax.set_yticklabels('')
    ax.set_xlim(-3,-1)
    ax.tick_params(direction='in', axis='y')
    ax.tick_params(direction='out', axis='x')
    labels = ['%.3f' %(10**x) for x in ax.get_xticks()]
    fmt_labels = [labels[i] if i % 2 == 0 else '' for i in range(len(labels))]
    ax.set_xticklabels(fmt_labels, fontsize=12)
    ax.set_xlabel("$M_{ej}$ [$M_\odot$]", fontsize=14)
    
    #Other subplots
    axs[0,1].set_visible(False)
    axs[0,2].set_visible(False)
    axs[1,2].set_visible(False)
    
    #add padding
    
    if outfile != '':
        fig.savefig(outfile)
        fig.savefig(outfile + '.jpg', rasterized=True)
    #fig.show()
    
    return
    
    
def make_plot_single(full_info, axs, plot_contours=True, hist_params={}):
    
    #VK vs LOGMASS
    ax = axs[2,0]
    if plot_contours:
        ax = make_contour(full_info['VK_LOGMASS'], ax)
    
    #VK vs LOGXLAN
    ax = axs[1,0]
    if plot_contours:
        ax = make_contour(full_info['VK_LOGXLAN'], ax)
    
    #LOGXLAN vs LOGMASS
    ax = axs[2,1]
    if plot_contours:
        ax = make_contour(full_info['LOGXLAN_LOGMASS'], ax)
    
    #VK
    ax = axs[0,0]
    ax = make_hist([full_info['VK_LOGXLAN']['Y'].T, np.mean(full_info['VK_LOGXLAN']['data'], axis=1)], ax, hist_params)
    
    #LOGXLAN
    ax = axs[1,1]
    ax = make_hist([full_info['VK_LOGXLAN']['X'], np.mean(full_info['VK_LOGXLAN']['data'], axis=0)], ax, hist_params)
    
    #LOGMASS
    ax = axs[2,2]
    ax = make_hist([full_info['VK_LOGMASS']['X'], np.mean(full_info['VK_LOGMASS']['data'], axis=0)], ax, hist_params)
    
    return axs
