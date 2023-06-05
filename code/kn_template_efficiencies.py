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

#read output of kn_properties.py
try:
    df = pd.read_csv('../events/%s/analysis/%s_kn_terse_cut_results_ml.csv' %(event_name, event_name))
    ml = True
except:
    df = pd.read_csv('../events/%s/analysis/%s_kn_terse_cut_results.csv' %(event_name, event_name))
    ml = False

#parse df to determine template efficiencies for each cut
def get_efficiency_df(cut, df):
    
    cuts, counts = np.unique(df['CUT'], return_counts=True)
    #print(cuts, counts)
    templates, totals = np.unique(df['SIM_TEMPLATE_INDEX'], return_counts=True)
    #print(len(templates), templates)#, totals)
    #simran = range(1,330)
    #for s,t in zip(simran,templates):
        #print(s,t)

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
                                  template_df['VK'].values[0],
                                  template_df['LOGXLAN'].values[0],
                                  template_df['LOGMASS'].values[0],
                                  eff,
                                  g_mean,
                                  r_mean,
                                  i_mean,
                                  z_mean])
        except:
            template_info.append([x, 
                                  df['VK'].values[df['SIM_TEMPLATE_INDEX'].values == x][0],
                                  df['LOGXLAN'].values[df['SIM_TEMPLATE_INDEX'].values == x][0],
                                  df['LOGMASS'].values[df['SIM_TEMPLATE_INDEX'].values == x][0],
                                  0.0, 
                                  99.0,
                                  99.0,
                                  99.0,
                                  99.0])

    template_info_cols = ['SNANA_INDEX', 'VK', 'LOGXLAN', 'LOGMASS', 'EFFICIENCY', 'PEAKMAG_g', 'PEAKMAG_r', 'PEAKMAG_i', 'PEAKMAG_z']
    output_df = pd.DataFrame(data=template_info, columns=template_info_cols)
    output_df.to_csv("../events/%s/analysis/%s_cut_%s_kn_efficiencies_table.csv" %(event_name, event_name, cut))
    
    #print(len(results))
    return np.array(results)

#make plot
def plot_efficiencies(effs, df, title=None, GW170817=True, outfile=None, skip_last=False):

    templates, totals = np.unique(df['SIM_TEMPLATE_INDEX'], return_counts=True)
    if skip_last:
        fig, axs = plt.subplots(1, len(np.unique(df['LOGXLAN'])) - 1, figsize=(18, 7.5), dpi=120)
    else:
        fig, axs = plt.subplots(1, len(np.unique(df['LOGXLAN'])), figsize=(18, 7.5), dpi=120)
    
    if title is not None:
        fig.suptitle(title, fontsize=28)
    
    logmasses, logmasses_counts = np.unique(df['LOGMASS'], return_counts=True)
    logxlans, logxlans_counts = np.unique(df['LOGXLAN'], return_counts=True)
    vks, vk_counts = np.unique(df['VK'], return_counts=True)
    
    counter = 0
    for logxlan in np.unique(df['LOGXLAN']):
        
        im_arr = np.ones((len(np.unique(df['LOGMASS'])), len(np.unique(df['VK']))))
        #im_arr = np.ones((len(np.unique(df['VK'])), len(np.unique(df['LOGMASS']))))
        vk_indices = np.arange(len(np.unique(df['VK'])))
        logmass_indices = np.arange(len(np.unique(df['LOGMASS'])))
        for vk_index in vk_indices:
            vk = vks[vk_index]
            for logmass_index in logmass_indices:
                logmass = logmasses[logmass_index]
                template = df[(df['LOGMASS'] == logmass) & (df['LOGXLAN'] == logxlan) & (df['VK'] == vk)]['SIM_TEMPLATE_INDEX'].values
                #print(template)
                if len(template) != 0:
                    #if template[0]<len(templates):
                    eff = effs[template[0] - 1]
                    #print(eff)#eff = effs[np.unique(template) - 1]
                else:
                    eff = 0.0
                    
                im_arr[logmass_index, vk_index] = eff
                
                
                if skip_last:
                    if eff < 0.3:
                        axs[counter].text(vk_index, logmass_index, '%.2f' %eff, ha="center", va="center", color="white", fontsize=16)
                    else:
                        axs[counter].text(vk_index, logmass_index, '%.2f' %eff, ha="center", va="center", color="black", fontsize=16)
                else:
                    if eff < 0.3:
                        axs[counter].text(vk_index, logmass_index, '%.2f' %eff, ha="center", va="center", color="white", fontsize=13)
                    else:
                        axs[counter].text(vk_index, logmass_index, '%.2f' %eff, ha="center", va="center", color="black", fontsize=13)

    
        axs[counter].imshow(im_arr, vmin=0, vmax=1, cmap='viridis', origin='lower')
        
        if counter > 0:
            axs[counter].set_yticklabels([])
            
        axs[counter].set_xticks(np.arange(len(vks)))
        if skip_last:
            axs[counter].set_xticklabels(['%.2f' %x for x in vks], fontsize=16)
        else:
            axs[counter].set_xticklabels(['%.2f' %x for x in vks], fontsize=12)
        
        #axs[counter].set_xlabel('EJECTA VELOCITY\nLOGXLAN = %.0f' %logxlans[counter], fontsize=20)
        if skip_last:
            axs[counter].set_xlabel('$v_{ej}$ [$c$]', fontsize=20)
            axs[counter].set_title('$\log (X_{lan})$ = %.0f' %logxlans[counter], fontsize=20)
        else:
            axs[counter].set_xlabel('$v_{ej}$ [$c$]', fontsize=16)
            axs[counter].set_title('$\log (X_{lan})$ = %.0f' %logxlans[counter], fontsize=16)
    
        counter += 1
        
        if skip_last:
            if counter == 5:
                break
    
    if skip_last:
        axs[0].set_ylabel("$M_{ej}$ [$M_\odot$]", fontsize=20)
        axs[0].set_yticklabels(['%.3f' %10**x for x in logmasses], fontsize=20)
    else:
        axs[0].set_ylabel("$M_{ej}$ [$M_\odot$]", fontsize=16)
        axs[0].set_yticklabels(['%.3f' %10**x for x in logmasses], fontsize=16)
    axs[0].set_yticks(np.arange(len(logmasses)))
    
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

    ## Color the missing template white and label N/A
    if not skip_last:
        vk_index = np.arange(len(vks))[np.where(vks == 0.3)][0]
        logmass_index = np.arange(len(logmasses))[np.where((logmasses >= np.log10(0.1) - 0.01) & (logmasses <= np.log10(0.1) + 0.01))][0]
        logxlan_index = np.arange(len(logxlans))[np.where((logxlans == -1))][0]
        axs[logxlan_index].add_patch(Rectangle((vk_index - 0.5, logmass_index - 0.5), 1, 1, fill=True, color='white'))
        axs[logxlan_index].text(vk_index, logmass_index, 'N/A', ha="center", va="center", color="black", fontsize=10)
        
    fig.tight_layout()
    
    if outfile is not None:
        plt.savefig(outfile + '.pdf', rasterized=True)


# Sanity Checks
logfile = open("../events/%s/analysis/kn_template_efficiencies.log" %event_name, 'w+')
lines = ["Tests of KN efficiencies\n\n",
         "print('Shape = ', df.shape)\n",
         'Shape = %s\n\n' %str(df.shape),
         "len(np.unique(df['SIM_TEMPLATE_INDEX']))\n"
         "%s\n\n" %str(len(np.unique(df['SIM_TEMPLATE_INDEX']))),
         "print(np.unique(df['CUT'], return_counts=True))\n",
         "%s\n\n" %str(np.unique(df['CUT'], return_counts=True)),
         "np.unique(df['VK'])\n",
         "%s\n\n" %str(np.unique(df['VK'])),
         "np.unique(df['LOGXLAN'])\n",
         "%s\n\n" %str(np.unique(df['LOGXLAN'])),
         "np.unique(df['LOGMASS'])\n"
         "%s\n\n" %str(np.unique(df['LOGMASS']))]
logfile.writelines(lines)
logfile.close()

# MAIN BODY
cuts = [x for x in np.unique(df['CUT'].values) if int(x) != -1]
cuts += [10 + np.max(cuts)]

efficiency_dfs = {'cut_%s' %cut: get_efficiency_df(cut, df) for cut in cuts}
#print(efficiency_dfs[cut_%s])

for cut in cuts:
    outfile_trimmed = "../events/%s/analysis/%s_cut_%s_kn_efficiencies_trimmed" %(event_name, event_name, cut)
    plot_efficiencies(efficiency_dfs['cut_%s' %cut], df, title=None, GW170817=True, outfile=outfile_trimmed, skip_last=False)

    outfile_full = "../events/%s/analysis/%s_cut_%s_kn_efficiencies_full" %(event_name, event_name, cut)
    plot_efficiencies(efficiency_dfs['cut_%s' %cut], df, title=None, GW170817=True, outfile=outfile_full, skip_last=False)

