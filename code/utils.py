# A module for quick calculations

from astropy.cosmology import z_at_value
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import easyaccess as ea
import numpy as np
import pandas as pd
from scipy.special import gammaln

def get_ligo_z_range(dist, sigma):

    #assume a cosmology
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    
    #calculate z range
    min_dist = (dist - sigma) * u.Mpc
    min_z = z_at_value(cosmo.luminosity_distance, min_dist, method='bounded')
    max_dist = (dist + sigma) * u.Mpc
    max_z = z_at_value(cosmo.luminosity_distance, max_dist, method='bounded')

    return round(min_z.value, 3), round(max_z.value, 3)


def MAGLIMIT_calculator(ZPT, PSF, SKYMAG, SNR):
    
    EA = 1.00
    OMEGA = 1.51
    eps = 10000.0  
    acc = 0.001      # required accuracy for convergence
    cnt, cnt_orig = 100, 100   
    t2  =  5.0 * np.log10(SNR/EA)
    tA  = ( OMEGA * PSF ) * ( OMEGA * PSF )  # area
    t1  = 2.5 * np.log10( tA )
    MLIM = 20.0     # very rough guess.
    
    while eps > acc and cnt > 0:

        if cnt == cnt_orig:
            #initial guess is with no signal-noise
            tMLIM  = 0.5 * ( ZPT + SKYMAG - t1 - t2 )
        else:
            tMLIM = MLIM # previous iteration

        arg = 0.4*(SKYMAG-tMLIM) # RK added this Feb 2011

        MLIM = 0.5*(tMLIM + ZPT - t2) - 1.25*np.log10(1.0 +  tA*10.0**-arg )
        
        eps = np.absolute( MLIM - tMLIM )
        
        cnt -= 1
        
    if cnt == 0:
        print("MLIM=%.3f value has not converged: eps=%f" %(MLIM, eps))
        
    return MLIM


def calc_expected_agn_using_mag_eff(event_info, eff_area, cut_results, flt, cut):
    #obtain detection efficiency as a function of magnitude
    ##extract peakmag
    cut_peak_mags = []
    total_peak_mags = []
    for snid, info in cut_results.items():
        fluxes = info['lightcurve'][info['lightcurve']['FLT'] == flt]['FLUXCAL'].values
        if len(fluxes) != 0:
            max_flux = np.max([float(x) for x in fluxes])
            if max_flux > 0:
                total_peak_mags.append(27.5 - 2.5 * np.log10(max_flux))
                if info['cut'] > cut or info['cut'] == -1:
                    cut_peak_mags.append(27.5 - 2.5* np.log10(max_flux))

    ##magnitude-binned calculation of efficiency
    min_mag = np.min(cut_peak_mags)
    max_mag = np.max(cut_peak_mags)
    binned_tot = np.array([float(x) for x in np.histogram(total_peak_mags, bins=10, range=(min_mag, max_mag))[0]])
    binned_det = np.array([float(x) for x in np.histogram(cut_peak_mags, bins=10,range=(min_mag,max_mag))[0]])
    binned_mag_eff = binned_det / binned_tot   # np.array([x / y for x in x, y in zip(binned_det, binned_tot)])
    if cut == 0: binned_mag_eff = np.ones_like(binned_tot)
    
    #connect to DB 
    conn = ea.connect(section='dessci')
    cursor = conn.cursor()

    # interpret magnitude limits 
    glim = event_info['LIM_MAG_g'].values[0] + event_info['LIM_MAG_g_std'].values[0]
    rlim = event_info['LIM_MAG_r'].values[0] + event_info['LIM_MAG_r_std'].values[0]
    ilim = event_info['LIM_MAG_i'].values[0] + event_info['LIM_MAG_i_std'].values[0]
    zlim = event_info['LIM_MAG_z'].values[0] + event_info['LIM_MAG_z_std'].values[0]

    #specify small regions of footproint to look at 
    footprint_regions = [[45.0, 45.2, -25.2, -25.0],
                         [43.0, 43.2, -26.2, -26.0],
                         [52.0, 52.2, -30.2, -30.0],
                         [56.0, 56.2, -25.2, -25.0],
                         [45.0, 45.2, -26.2, -26.0],
                         [52.0, 52.2, -25.2, -25.0],
                         [56.0, 56.2, -30.2, -30.0],
                         [43.0, 43.2, -25.2, -25.0],
                         [43.0, 43.2, -30.2, -30.0],
                         [56.0, 56.2, -25.2, -25.0]]

    #calulate the density in each region and average them 
    scaled_num_galaxies = []
    for region in footprint_regions:
        ra_min = region[0]
        ra_max = region[1]
        dec_min = region[2]
        dec_max = region[3]

        area = (ra_max - ra_min) * (np.sin(dec_max * np.pi / 180.0) - np.sin(dec_min * np.pi / 180.0)) * 180.0 / np.pi

        query = ["SELECT SOF_CM_MAG_%s FROM Y3_GOLD_2_2" %flt.upper(),
             "WHERE RA < ", str(ra_max), "and RA > ", str(ra_min),
             "and DEC < ", str(dec_max), "and DEC > ", str(dec_min),
             "and EXTENDED_CLASS_MASH_SOF = 3 ",
             "and FLAGS_FOOTPRINT = 1 ",
             "and FLAGS_FOREGROUND = 0 ",
             "and bitand(FLAGS_GOLD, 62) = 0 ",
             "and SOF_CM_MAG_I between 16 and 26 ",
             "and (SOF_CM_MAG_G < ", str(glim),
             "or SOF_CM_MAG_R < ", str(rlim),
             "or SOF_CM_MAG_I < ", str(ilim),
             "or SOF_CM_MAG_Z < ", str(zlim),
             ")"]

        #print(' '.join(query))  
        QQ = cursor.execute(' '.join(query))
        rows = cursor.fetchall()
        
        mag_df = pd.DataFrame(data=rows, columns=['%s_mag' %flt])

        binned_mag_counts = np.array([float(x) for x in np.histogram(mag_df[['%s_mag' %flt]].values, bins=10, range=(min_mag, max_mag))[0]])
        
        num_galaxies = np.dot(binned_mag_counts, binned_mag_eff)
        scaled_num_galaxies.append(num_galaxies / area * eff_area)
        
    mean_num_galaxies = np.mean(scaled_num_galaxies)
    std_num_galaxies = np.std(scaled_num_galaxies) / len(scaled_num_galaxies)**2

    agn_rate, agn_plus_unc, agn_minus_unc = 0.0016, 0.0006, 0.0006

    num_agn = mean_num_galaxies * agn_rate
    num_agn_plus_unc = num_agn * np.sqrt((agn_plus_unc / agn_rate) ** 2 + (std_num_galaxies / mean_num_galaxies) ** 2)
    num_agn_minus_unc = num_agn * np.sqrt((agn_minus_unc / agn_rate) ** 2 + (std_num_galaxies / mean_num_galaxies) ** 2)

    return num_agn, [num_agn_plus_unc, num_agn_minus_unc]



def calc_expected_agn(event_info, eff_area):
    #connect to DB
    conn = ea.connect(section='dessci')
    cursor = conn.cursor()

    # interpret magnitude limits
    glim = event_info['LIM_MAG_g'].values[0] + event_info['LIM_MAG_g_std'].values[0]
    rlim = event_info['LIM_MAG_r'].values[0] + event_info['LIM_MAG_r_std'].values[0]
    ilim = event_info['LIM_MAG_i'].values[0] + event_info['LIM_MAG_i_std'].values[0]
    zlim = event_info['LIM_MAG_z'].values[0] + event_info['LIM_MAG_z_std'].values[0]

    #specify small regions of footproint to look at
    footprint_regions = [[45.0, 45.2, -25.2, -25.0],
                         [43.0, 43.2, -26.2, -26.0],
                         [52.0, 52.2, -30.2, -30.0],
                         [56.0, 56.2, -25.2, -25.0],
                         [45.0, 45.2, -26.2, -26.0],
                         [52.0, 52.2, -25.2, -25.0],
                         [56.0, 56.2, -30.2, -30.0],
                         [43.0, 43.2, -25.2, -25.0],
                         [43.0, 43.2, -30.2, -30.0],
                         [56.0, 56.2, -25.2, -25.0]]

    #calulate the density in each region and average them
    scaled_num_galaxies = []
    for region in footprint_regions:
        ra_min = region[0]
        ra_max = region[1]
        dec_min = region[2]
        dec_max = region[3]

        area = (ra_max - ra_min) * (np.sin(dec_max * np.pi / 180.0) - np.sin(dec_min * np.pi / 180.0)) * 180.0 / np.pi

        query = ["SELECT COUNT(*) FROM Y3_GOLD_2_2", 
             "WHERE RA < ", str(ra_max), "and RA > ", str(ra_min),
             "and DEC < ", str(dec_max), "and DEC > ", str(dec_min),
             "and EXTENDED_CLASS_MASH_SOF = 3 ",
             "and FLAGS_FOOTPRINT = 1 ",
             "and FLAGS_FOREGROUND = 0 ",
             "and bitand(FLAGS_GOLD, 62) = 0 ",
             "and SOF_CM_MAG_I between 16 and 26 ",
             "and (SOF_CM_MAG_G < ", str(glim),  
             "or SOF_CM_MAG_R < ", str(rlim), 
             "or SOF_CM_MAG_I < ", str(ilim),
             "or SOF_CM_MAG_Z < ", str(zlim), 
             ")"]
        
        #print(' '.join(query))
        
        QQ = cursor.execute(' '.join(query))
        rows = cursor.fetchall()

        num_galaxies = float(rows[0][0])
        scaled_num_galaxies.append(num_galaxies / area * eff_area)

    mean_num_galaxies = np.mean(scaled_num_galaxies)
    std_num_galaxies = np.std(scaled_num_galaxies) / len(scaled_num_galaxies)

    #incorporate measured agn fraction from https://kb.osu.edu/bitstream/handle/1811/84811/1/Michael_Macuga_Honors_Thesis.pdf 
    #agn_rate, agn_plus_unc, agn_minus_unc = 0.020, 0.026, 0.013

    #incorporate measured agn fraction from https://www.researchgate.net/publication/...
    #45911158_The_Field_X-ray_AGN_Fraction_to_z07_from_the_Chandra_Multiwavelength_Project_and_the_Sloan_Digital_Sky_Survey
    agn_rate, agn_plus_unc, agn_minus_unc = 0.0016, 0.0006, 0.0006
    
    num_agn = mean_num_galaxies * agn_rate
    num_agn_plus_unc = num_agn * np.sqrt((agn_plus_unc / agn_rate) ** 2 + (std_num_galaxies / mean_num_galaxies) ** 2)
    num_agn_minus_unc = num_agn * np.sqrt((agn_minus_unc / agn_rate) ** 2 + (std_num_galaxies / mean_num_galaxies) ** 2)

    return num_agn, [num_agn_plus_unc, num_agn_minus_unc]



def gammalnStirling(z):
    """
    Uses Stirling's approximation for the log-gamma function suitable for large arguments.
    """
    return (0.5 * (np.log(2. * np.pi) - np.log(z))) + (z * (np.log(z + (1. / ((12. * z) - (1. / (10. * z))))) - 1.))


def confidenceInterval(n, k, alpha=0.68, errorbar=True):
    """
    Given n tests and k successes, return efficiency and confidence interval.
    Source: https://github.com/DarkEnergySurvey/ugali/blob/master/ugali/utils/bayesian_efficiency.py
    """
    try:
        e = float(k) / float(n)
    except ZeroDivisionError:
        return np.nan, [np.nan, np.nan]

    bins = 1000001
    dx = 1. / bins

    efficiency = np.linspace(0, 1, bins)

    # MODIFIED FOR LARGE NUMBERS
    if n + 2 > 1000:
        a = gammalnStirling(n + 2)
    else:
        a = gammaln(n + 2)
    if k + 1 > 1000:
        b = gammalnStirling(k + 1)
    else:
        b = gammaln(k + 1)
    if n - k + 1 > 1000:
        c = gammalnStirling(n - k + 1)
    else:
        c = gammaln(n - k + 1)

    if k == 0:
        p = np.concatenate([[np.exp(a - b - c)],
                               np.exp(a - b - c + (k * np.log(efficiency[1: -1])) + (n - k) * np.log(1. - efficiency[1: -1])),
                               [0.]])
    elif k == n:
        p = np.concatenate([[0.],
                               np.exp(a - b - c + (k * np.log(efficiency[1: -1])) + (n - k) * np.log(1. - efficiency[1: -1])),
                               [np.exp(a - b - c)]])
    else:
        p = np.concatenate([[0.],
                               np.exp(a - b - c + (k * np.log(efficiency[1: -1])) + (n - k) * np.log(1. - efficiency[1: -1])),
                               [0.]])

    i = np.argsort(p)[::-1]
    p_i = np.take(p, i)

    s = i[np.cumsum(p_i * dx) < alpha]
    
    try:
        low = min(np.min(s) * dx, e)
        high = max(np.max(s) * dx, e)
    except:
        low = 0.0
        high = 0.0

    if not errorbar:
        return e, [low, high]
    else:
        return e, [e - low, high - e]


def open_dat(dat_file):
    infile = open(dat_file, 'r')
    lines = infile.readlines()
    infile.close()
    return lines

def get_terse_lc(lines):

    columns = [y for y in [x.split(' ') for x in lines if x[0:8] == 'VARLIST:'][0] if y != ''][1:-1]
    data = [[y for y in x.split(' ') if y != ''][1:-1] for x in lines if x[0:4] == 'OBS:']
    df = pd.DataFrame(data=data, columns=columns)
    
    for col in columns:
        if col != 'FLT' and col != 'FIELD':
            df[col] = pd.to_numeric(df[col])

    return df

def get_meta_data(lines):

    snid = [y for y in [x for x in lines if x[0:5] == 'SNID:'][0].split(' ') if y != ''][1][0:-1]
    ra = float([y for y in [x for x in lines if x[0:3] == 'RA:'][0].split(' ') if y != ''][1])
    dec = float([y for y in [x for x in lines if x[0:3] == 'DEC'][0].split(' ') if y != ''][1])

    return {'SNID': snid, 'RA': ra, 'DEC': dec}
