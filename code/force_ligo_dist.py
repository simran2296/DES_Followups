# A module to fit the LIGO distance posterior

from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
import astropy.units as u
import numpy as np
from scipy.optimize import curve_fit

def convert_to_z(d, cosmo):
    return z_at_value(cosmo.luminosity_distance, d * u.Mpc, method='bounded')

def double_poly_fit(x, b0, b1, b2, b3, c0, c1, c2, c3):
    return (b0 + b1*x + b2*x**2 + b3*x**3) * (c0 + c1*x + c2*x**2 + c3*x**3)

def fit(avg, std, H0=70, Tcmb0=2.725, Om0=0.3):
    
    cosmo = FlatLambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Tcmb0=Tcmb0 * u.K, Om0=Om0)

    ligo_avg_z = convert_to_z(avg, cosmo)
    ligo_std_z = convert_to_z(std, cosmo)

    z_posterior = np.random.normal(loc=ligo_avg_z, scale=ligo_std_z, size=10000)
    z_posterior = z_posterior[z_posterior > 0.0]

    counts, edges = np.histogram(z_posterior, bins=100, density=True)
    midpoints = [0.5 * edges[ii] + 0.5 * edges[ii + 1] for ii in range(len(edges) -1)]

    attempt = 0
    while attempt < 11:
        
        try:
            popt, pcov = curve_fit(double_poly_fit, midpoints, counts, maxfev=10000)
            attempt = 50
        except:
            print("Error in curve_fit. Retrying... (%i / 10)" %attempt)
            np.random.seed(attempt)
            attempt += 1

    if attempt == 11:
        print("Error in curve fit of avg = %s, std = %s. Defaulting to LIGO_mean +/- 2 sigma" %(avg, std))
        
        popt, pcov = curve_fit(double_poly_fit, midpoints, counts, maxfev=10000)
        assert False

    res = np.array([double_poly_fit(x, *popt) for x in midpoints])
    
    return popt, midpoints, counts, res

def get_zmin_zmax(resid, clean_midpoints, resid_thresh=2.0, z_thresh=0.005):
    #z min
    left_resid = list(resid[clean_midpoints < np.median(clean_midpoints)])
    left_resid.reverse()
    left_midpoints = list(clean_midpoints[clean_midpoints < np.median(clean_midpoints)])
    left_midpoints.reverse()
    z_min = np.median(clean_midpoints)
    
    zmin_found = False
    z_prev = z_min + 0.0
    for z, res in zip(left_midpoints, left_resid):
        if res > resid_thresh:
            z_min = z
            zmin_found = True
            break
        if z - z_prev > z_thresh:
            z_min = z_prev
            zmin_found = True
            break
        z_prev = z + 0.0
        
    if not zmin_found:
        z_min = 0.003

    if z_min < 0.003:
        z_min = 0.003
        
    #z max
    right_resid = list(resid[clean_midpoints > np.median(clean_midpoints)])
    right_midpoints = list(clean_midpoints[clean_midpoints > np.median(clean_midpoints)])
    z_max = np.median(clean_midpoints)
    
    zmax_found = False
    z_prev = z_max + 0.0
    for z, res in zip(right_midpoints, right_resid):
        if res > resid_thresh:
            z_max = z
            zmax_found = True
            break
        if z - z_prev > z_thresh:
            z_max = z_prev
            zmax_found = True
            break
        z_prev = z + 0.0
        
    if not zmax_found:
        z_max = np.max(clean_midpoints)
        
    return z_min, z_max, zmin_found, zmax_found

def check(event_name, midpoints, counts, res, popt, obj):
    test_arr = np.array(res)
    midpoint_arr = np.array(midpoints)
    cond = (test_arr >= 0.0)
    clean_test = test_arr[cond]
    clean_midpoints = midpoint_arr[cond]
    clean_counts = counts[cond]
    resid = clean_counts / clean_test

    zmin, zmax, zmin_found, zmax_found = get_zmin_zmax(resid, clean_midpoints)
    write_report(event_name, zmin, zmax, zmin_found, zmax_found, popt, obj)
    
    return zmin, zmax

def convert_to_snana_float(num):
    return '%.3E' %num

def write_report(event_name, zmin, zmax, zmin_found, zmax_found, popt, obj):
    log_dir = '../events/%s/logs/' %event_name

    outlines = ['LIGO DISTANCE POSTERIOR FIT REPORT',
                'zmin = %.4f' %zmin,
                'zmax = %.4f' %zmax,
                'zmin_found = %i' %int(zmin_found),
                'zmax_found = %i' %int(zmax_found),
                'poly1 = %s' %' '.join([convert_to_snana_float(x) for x in popt[:4]]),
                'poly2 = %s' %' '.join([convert_to_snana_float(x) for x in popt[4:]])]

    stream = open(log_dir + 'force_dist_%s.log' %obj, 'w+')
    stream.writelines([x + '\n' for x in outlines])
    stream.close()

    return

def update_snana(obj_filename, event_name, popt, zmin, zmax):
    simgen_dir = '../events/%s/sim_gen/' %event_name
    filename = simgen_dir + obj_filename

    stream = open(filename, 'r')
    lines = stream.readlines()
    stream.close()

    outlines = []
    for line in lines:

        if line[0:5] == 'DNDZ:':
            outlines.append('DNDZ: ZPOLY  %s\n' %' '.join([convert_to_snana_float(x) for x in popt[:4]]))
            outlines.append('DNDZ_ZPOLY_REWGT:  %s\n' %' '.join([convert_to_snana_float(x) for x in popt[4:]]))

        elif line[0:18] == 'GENRANGE_REDSHIFT:':
            outlines.append('GENRANGE_REDSHIFT:  %.4f  %.4f\n' %(zmin, zmax))
            
        else:
            outlines.append(line)

    stream = open(filename, 'w+')
    stream.writelines(outlines)
    stream.close()

    return
    
    

def run(obj_filename, event_name, avg, std):
    popt, midpoints, counts, res = fit(avg, std)

    obj = obj_filename.split('.')[0]
    
    zmin, zmax = check(event_name, midpoints, counts, res, popt, obj)
    
    update_snana(obj_filename, event_name, popt, zmin, zmax)
    
    return
