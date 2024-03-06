# Plot KN efficiencies

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from matplotlib.patches import Rectangle
from matplotlib import rcParams
rcParams['font.family'] = 'serif'

import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
mpl.use('Agg')
import matplotlib.pyplot as plt

event_name = sys.argv[1]

#read output of BBH_properties.py
try:
    df = pd.read_csv('../events/%s/BBH_analysis/%s_BBH_terse_cut_results_ml.csv' %(event_name, event_name))
    ml = True
except:
    df = pd.read_csv('../events/%s/BBH_analysis/%s_BBH_terse_cut_results.csv' %(event_name, event_name))
    ml = False

#parse df to determine template efficiencies for each cut
def get_efficiency_df(cut, df):
    
    cuts, counts = np.unique(df['CUT'], return_counts=True)
    print(cuts, counts)
    templates, totals = np.unique(df['SIM_TEMPLATE_INDEX'], return_counts=True)
    print(len(templates), templates, totals)
    

    cut_df = df.iloc[np.where((df['CUT'].values > cut) | (df['CUT'].values == -1))]
    
    results = []
    template_info = []
    for x in sorted(templates):
        #index = templates.tolist().index(x)
        template_df = cut_df[cut_df['SIM_TEMPLATE_INDEX'] == x]
        eff = float(template_df.shape[0]) / float(totals[x - 1]) #float(totals[index])
        results.append(eff)
        try:
            if len(template_df['PEAKMAG_g'].values[template_df['PEAKMAG_g'].values < 27.5]) > 0:
                g_mean = np.mean(template_df['PEAKMAG_g'].values[template_df['PEAKMAG_g'].values < 27.5])
            else:
                g_mean = 99.0
            if len(template_df['PEAKMAG_r'].values[template_df['PEAKMAG_r'].values < 27.5]) > 0:
                r_mean = np.mean(template_df['PEAKMAG_r'].values[template_df['PEAKMAG_r'].values < 27.5])
            else:
                r_mean = 99.0
            if len(template_df['PEAKMAG_i'].values[template_df['PEAKMAG_i'].values < 27.5]) > 0:
                i_mean = np.mean(template_df['PEAKMAG_i'].values[template_df['PEAKMAG_i'].values < 27.5])
            else:
                i_mean = 99.0
            if len(template_df['PEAKMAG_z'].values[template_df['PEAKMAG_z'].values < 27.5]) > 0:
                z_mean = np.mean(template_df['PEAKMAG_z'].values[template_df['PEAKMAG_z'].values < 27.5])
            else:
                z_mean = 99.0

            template_info.append([x,
                                  template_df['rise_id'].values[0],
                                  template_df['fall_id'].values[0],
                                  template_df['brightness_param'].values[0],
                                  eff,
                                  g_mean,
                                  r_mean,
                                  i_mean,
                                  z_mean])
        except:
            template_info.append([x, 
                                  df['rise_id'].values[df['SIM_TEMPLATE_INDEX'].values == x][0],
                                  df['fall_id'].values[df['SIM_TEMPLATE_INDEX'].values == x][0],
                                  df['brightness_param'].values[df['SIM_TEMPLATE_INDEX'].values == x][0],
                                  0.0, 
                                  99.0,
                                  99.0,
                                  99.0,
                                  99.0])

    template_info_cols = ['SNANA_INDEX', 'rise_index', 'fall_index', 'brightness_param', 'EFFICIENCY', 'PEAKMAG_g', 'PEAKMAG_r', 'PEAKMAG_i', 'PEAKMAG_z']
    output_df = pd.DataFrame(data=template_info, columns=template_info_cols)
    output_df.to_csv("../events/%s/BBH_analysis/%s_cut_%s_BBH_efficiencies_table.csv" %(event_name, event_name, cut))
    
    #print(len(results))
    return np.array(results)

#make plot
def plot_efficiencies(effs, df, title=None, GW170817=True, outfile=None, skip_last=False):
    
    eff_list_1 = []
    eff_list_2 = []

    templates, totals = np.unique(df['SIM_TEMPLATE_INDEX'], return_counts=True)
    if skip_last:
        fig, axs = plt.subplots(1, len(np.unique(df['brightness_param'])) - 1, figsize=(18, 7.5), dpi=120, squeeze=False)
    else:
        fig, axs = plt.subplots(1, len(np.unique(df['brightness_param'])), figsize=(18, 7.5), dpi=120, squeeze=False)
    
    if title is not None:
        fig.suptitle(title, fontsize=28)
    
    #logmasses, logmasses_counts = np.unique(df['LOGM'], return_counts=True)
    #logxlans, logxlans_counts = np.unique(df['LOGXLAN'], return_counts=True)
    #vks, vk_counts = np.unique(df['VK'], return_counts=True)
    rise_indices, rise_indices_counts = np.unique(df['rise_id'], return_counts=True)
    fall_indices, fall_indices_counts = np.unique(df['fall_id'], return_counts=True)
    brightness_params, brightness_params_counts = np.unique(df['brightness_param'], return_counts=True)

    counter = 0
    for param in np.unique(df['brightness_param']):
        
        im_arr = np.ones((len(np.unique(df['rise_id'])), len(np.unique(df['fall_id']))))
        #im_arr = np.ones((len(np.unique(df['VK'])), len(np.unique(df['LOGMASS']))))
        fall_indices_1 = np.arange(len(np.unique(df['fall_id'])))
        rise_indices_1 = np.arange(len(np.unique(df['rise_id'])))
        for fall_idx in fall_indices_1:
            fall = fall_indices[fall_idx]
            for rise_idx in rise_indices_1:
                rise = rise_indices[rise_idx]
                template = df[(df['rise_id'] == rise) & (df['brightness_param'] == param) & (df['fall_id'] == fall)]['SIM_TEMPLATE_INDEX'].values
                #print(template)
                if len(template) != 0:
                    #if template[0]<len(templates):
                    eff = effs[template[0] - 1]
                    #print(eff)#eff = effs[np.unique(template) - 1]
                else:
                    eff = 0.0
                    
                im_arr[rise_idx, fall_idx] = eff
                
                
                if skip_last:
                    if eff < 0.3:
                        axs[counter, 0].text(fall_idx, rise_idx, '%.2f' %eff, ha="center", va="center", color="white", fontsize=16)
                    else:
                        axs[counter, 0].text(fall_idx, rise_idx, '%.2f' %eff, ha="center", va="center", color="black", fontsize=16)
                else:
                    if eff < 0.3:
                        axs[counter, 0].text(fall_idx, rise_idx, '%.2f' %eff, ha="center", va="center", color="white", fontsize=13)
                    else:
                        axs[counter, 0].text(fall_idx, rise_idx, '%.2f' %eff, ha="center", va="center", color="black", fontsize=13)

                if eff > 0.9:
                    eff_list_1.append(eff)
                    #print(eff)

                if eff > 0.75:
                    eff_list_2.append(eff)
    
        axs[counter, 0].imshow(im_arr, vmin=0, vmax=1, cmap='viridis', origin='lower')
        
        if counter > 0:
            axs[counter].set_yticklabels([])
            
        axs[counter, 0].set_xticks(np.arange(len(fall_indices)))
        if skip_last:
            axs[counter, 0].set_xticklabels(['%.2f' %x for x in fall_indices], fontsize=16)
        else:
            axs[counter, 0].set_xticklabels(['%.2f' %x for x in fall_indices], fontsize=12)
        
        #axs[counter].set_xlabel('EJECTA VELOCITY\nLOGXLAN = %.0f' %logxlans[counter], fontsize=20)
        if skip_last:
            axs[counter, 0].set_xlabel('fall_index [$mags/day$]', fontsize=20)
            axs[counter, 0].set_title('brightness_param = %.0f [$ergs/cm^2/s$]' %brightness_params[counter], fontsize=20)
        else:
            axs[counter, 0].set_xlabel('fall_index [$mags/day$]', fontsize=16)
            axs[counter, 0].set_title('brightness_param = %.0f [$ergs/cm^2/s$]' %brightness_params[counter], fontsize=16)
    
        counter += 1
        
        if skip_last:
            if counter == 5:
                break
    
    if skip_last:
        axs[0,0].set_ylabel("rise_index [$days$]", fontsize=20)
        axs[0,0].set_yticklabels(['%.3f' %10**x for x in rise_indices], fontsize=20)
    else:
        axs[0,0].set_ylabel("rise_index [$days$]", fontsize=16)
        axs[0,0].set_yticklabels(['%.3f' %10**x for x in rise_indices], fontsize=16)
    axs[0,0].set_yticks(np.arange(len(rise_indices)))
    
    '''
    ## Outline GW170817 components in blue and red
    if GW170817:
        #blue: 0.025 Msun, 0.3c, log X = -4
        vk_index = np.arange(len(vks))[np.where(vks == 0.3)][0]
        logmass_index = np.arange(len(logmasses))[np.where((logmasses >= np.log10(0.025) - 0.01) & (logmasses <= np.log10(0.025) + 0.01))][0]
        logxlan_index = np.arange(len(logxlans))[np.where((logxlans == -4))][0]
        axs[logxlan_index].add_patch(Rectangle((vk_index - 0.5, logmass_index - 0.5), 1, 1, fill=False, edgecolor='blue', lw=3))
        #red: 0.04 Msun, 0.1c, log X = -2
        vk_index = np.arange(len(vks))[np.where(vks == 0.1)][0]
        logmass_index = np.arange(len(logmasses))[np.where((logmasses >= np.log10(0.04) - 0.01) & (logmasses <= np.log10(0.04) + 0.01))][0]
        logxlan_index = np.arange(len(logxlans))[np.where((logxlans == -2))][0]
        axs[logxlan_index].add_patch(Rectangle((vk_index - 0.5, logmass_index - 0.5), 1, 1, fill=False, edgecolor='red', lw=3))
    '''

    '''
    ## Color the missing template white and label N/A
    if not skip_last:
        vk_index = np.arange(len(vks))[np.where(vks == 0.3)][0]
        logmass_index = np.arange(len(logmasses))[np.where((logmasses >= np.log10(0.1) - 0.01) & (logmasses <= np.log10(0.1) + 0.01))][0]
        logxlan_index = np.arange(len(logxlans))[np.where((logxlans == -1))][0]
        axs[logxlan_index].add_patch(Rectangle((vk_index - 0.5, logmass_index - 0.5), 1, 1, fill=True, color='white'))
        axs[logxlan_index].text(vk_index, logmass_index, 'N/A', ha="center", va="center", color="black", fontsize=10)
    ''' 
    fig.tight_layout()
    
    if outfile is not None:
        plt.savefig(outfile + '.pdf', rasterized=True)
        plt.savefig(outfile + '.jpg', rasterized=True)


    good_eff_1 = len(eff_list_1)
    good_eff_2 = len(eff_list_2)
    #frac_eff = good_eff/329
    print('Fraction of Templates with efficiency greater than 90% = ', good_eff_1)    
    print('Fraction of Templates with efficiency greater than 75% = ', good_eff_2)


# Sanity Checks
logfile = open("../events/%s/BBH_analysis/BBH_template_efficiencies.log" %event_name, 'w+')
lines = ["Tests of BBH efficiencies\n\n",
         "print('Shape = ', df.shape)\n",
         'Shape = %s\n\n' %str(df.shape),
         "len(np.unique(df['SIM_TEMPLATE_INDEX']))\n"
         "%s\n\n" %str(len(np.unique(df['SIM_TEMPLATE_INDEX']))),
         "print(np.unique(df['CUT'], return_counts=True))\n",
         "%s\n\n" %str(np.unique(df['CUT'], return_counts=True)),
         "np.unique(df['fall_id'])\n",
         "%s\n\n" %str(np.unique(df['fall_id'])),
         "np.unique(df['brightness_param'])\n",
         "%s\n\n" %str(np.unique(df['brightness_param'])),
         "np.unique(df['rise_id'])\n"
         "%s\n\n" %str(np.unique(df['rise_id']))]
logfile.writelines(lines)
logfile.close()

# MAIN BODY
cuts = [x for x in np.unique(df['CUT'].values) if int(x) != -1]
cuts += [10 + np.max(cuts)]

efficiency_dfs = {'cut_%s' %cut: get_efficiency_df(cut, df) for cut in cuts}
#print(efficiency_dfs[cut_%s])

for cut in cuts:
    outfile_trimmed = "../events/%s/BBH_analysis/%s_cut_%s_BBH_efficiencies_trimmed" %(event_name, event_name, cut)
    plot_efficiencies(efficiency_dfs['cut_%s' %cut], df, title=None, GW170817=True, outfile=outfile_trimmed, skip_last=False)

    outfile_full = "../events/%s/BBH_analysis/%s_cut_%s_BBH_efficiencies_full" %(event_name, event_name, cut)
    plot_efficiencies(efficiency_dfs['cut_%s' %cut], df, title=None, GW170817=True, outfile=outfile_full, skip_last=False)

