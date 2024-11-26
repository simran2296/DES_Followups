#!/usr/bin/env python
"""%prog [options] -o OUTFILE

Written by Kyle Barberry
Hacked a bit by Zoheyr Doctor (Jan 2018)

Get observing conditions for SN coadd images from the database.
Currently these are written to an SNANA "SIMLIB" file.

By default, positions are chosen randomly from an area that entirely encloses
all the SN fields. For each position, the image WCS information from the
'diff_nitecmb_diff' or 'diff_single_diff' images is used to determine
whether each image covers the position. ('diff_nitecmb_diff' is used for deep
fields g, r, i, z and shallow fields z; 'diff_single_diff' is used for shallow
fields g, r, i). At the end, an "effective area" is reported. This is the area
of sky that has at least one observation. It can be thought of as the 
area from which the positions *in the output file* were selected
(because positions are not written to the output file unless  they have
at least one observation). It is determined statistically, by multiplying the
total area (entirely enclosing all the SN fields) by the fraction of positions
that have at least one observation.

Note that masking is ignored. If a position lies anywhere on the
image, it is considered as being observed in the image, even if turns
out to be on a masked pixel.  This results in overestimation of
observations.
"""

import random
import os
import sys
import math
import time
import csv
from optparse import OptionParser
try:
    from collections import OrderedDict as odict
except ImportError:
    odict = dict
import numpy as np
import easyaccess as ea
import pandas as pd

master_head_template = """\
TELESCOPE: CTIO
SURVEY: DES      FILTERS: griz
USER: username    HOST: hostname
COMMENT: {0}
PIXSIZE: 0.27
PSF_UNIT: ARCSEC_FWHM
#--------------------------------------------------------------------------
"""

head_template = """\
    
# --------------------------------------------
LIBID: {0:d}
RA: {1:.6f}    DECL: {2:.6f}   NOBS: {3:d}
MWEBV: {4:.3f}   PIXSIZE: {5:.3f}
"""
field_template = """\
FIELD: {0:s}  # CCDS: {1:s}
TEMPLATE_ZPT: {2:s}
TEMPLATE_SKYSIG: {3:s}
"""

head_comment = """\
#                           CCD  CCD           PSF1 PSF2 PSF2/1
#     MJD      IDEXPT  FLT GAIN NOISE  SKYSIG  (pixels)  RATIO ZPTAVG ZPTSIG    MAG
"""    

line_template = ('{0}: {1:9.3f} {2:10d} {3:1s} {4:5.2f} {5:5.2f} {6:7.2f}  '
                 '{7:4.2f} {8:4.2f} {9:5.3f} {10:6.2f} {11:6.3f} {12:7.3f}\n')

cache_filename = 'des-obsinfo-cache.npy'

parser=OptionParser(__doc__)
parser.add_option('--infile', default=None,
                  help='If specified, use positions read from this file. '
                  'Format: csv (comma-separated values) with csv header '
                  'line. "RA" and "DEC" (case-insensitive) must '
                  'appear in header line. Additional fields are '
                  'written to final SIMLIB file.')
parser.add_option('-o', '--outfile', default=None,
                  help='name of file to write output to [REQUIRED]')
parser.add_option('--dbname', default='desoper',
                  help='name of database to connect to [default=desoper]')
parser.add_option('-n', type='int', default=100,
                  help='number of entries to generate if infile not given '
                  '(default=100)')
parser.add_option('-c', '--cache', action='store_true', default=False,
                  help='cache query results in ./{0}'.format(cache_filename))
parser.add_option('--zpmin', type='float', default=False,
                  help='minimum zeropoint')
parser.add_option('--exptable',default=None,
                  help='name of expnum table which is just a vertical '
                  'list of expnums in a text file')

#RM hack to include bounds as command-line arguments
parser.add_option('--min_ra', default=0.0, help="Minimum RA of bounding box for observations")
parser.add_option('--max_ra', default=360.0, help="Maximum RA of bounding box for observations")
parser.add_option('--min_dec', default=-90.0, help="Minimum Dec of bounding box for observations")
parser.add_option('--max_dec', default=90.0, help="Maximum Dec of bounding box for observations")
options, args = parser.parse_args(sys.argv[1:])
min_ra = float(options.min_ra)
max_ra = float(options.max_ra)
min_dec = float(options.min_dec)
max_dec = float(options.max_dec)
                                            
# Definitions for randomly selection positions
#bounds = {'C': (51.2, 55.6, -30.2, -26.0),
#          'X': (33.3, 37.6,  -7.5,  -3.5),
#          'E': ( 6.2, 11.2, -45.1, -41.9),
#          'S': (40.0, 44.0,  -2.0,   1.2)}
#bounds = {'GW':(30,55,-57,-25)}
#bounds = {'1':(75,79,3,7)}       #ICECUBE Event 1
#bounds = {'2':(338, 342, 5,9)}   #ICECUBE Event 2
#bounds = {'GW': (75.0, 104.0, -43.0, -22.0)} #190510

bounds = {'GW': (min_ra, max_ra, min_dec, max_dec)}


# Area of each field
d2r = math.pi/180.  # radians per degree
r2d = 1./d2r
areas = {}
for key, (ra_min, ra_max, dec_min, dec_max) in bounds.items():
    areas[key] = ((math.sin(dec_max * d2r) - math.sin(dec_min * d2r)) *
                  (ra_max - ra_min) * d2r) * r2d**2

# weight of each field (percentage of total area)
totarea = sum(areas.values())
weights = dict((key, val / totarea) for key, val in areas.items())

# "cdf" of field weights
cdf = []
cumsum = 0.
for f, w in weights.items():
    cumsum += w
    cdf.append((f, cumsum))

def get_random_position():
    """Get a random RA, Dec position from the fields defined above.

    We could choose a random point from the entire sphere, which would
    be simpler and possibly more accurate, but only 0.1% of the sphere
    falls in our fields so it would take about 1000 tries to  get a
    single point in our fields. This is only slightly more complex.
    """
    r = random.random()
    i = 0
    while cdf[i][1] < r:
        i += 1
    f = cdf[i][0]
    ra = bounds[f][0] + random.random() * (bounds[f][1] - bounds[f][0])
    dec = r2d * math.asin(math.sin(bounds[f][2]*d2r) +
                          random.random() *
                          (math.sin(bounds[f][3]*d2r) -
                           math.sin(bounds[f][2]*d2r)))
    return ra, dec


def point_line_side(pointx, pointy, x1, y1, x2, y2):
    """Determine what side of a line a point is on.

    Returns positive number if point is on left side of the line from
    (x1,y1) to (x2,y2).
    Returns 0 if point is on the line.
    Returns negative number if point is on the right side.
    """
    return (x2-x1) * (pointy-y1) - (y2-y1) * (pointx-x1)


def point_within_quad(pointx, pointy, x1, y1, x2, y2, x3, y3, x4, y4):
    """Determine if an (x, y) point lies within a quadrilateral.

    (x1, y1) is the first corner of the quadrilateral, (x2,y2) is the adjacent
    corner proceeding counterclockwise (CCW), etc. Note that the direction
    is significant.
    """
    return ((point_line_side(pointx, pointy, x1, y1, x2, y2) >= 0.) &
            (point_line_side(pointx, pointy, x2, y2, x3, y3) >= 0.) &
            (point_line_side(pointx, pointy, x3, y3, x4, y4) >= 0.) &
            (point_line_side(pointx, pointy, x4, y4, x1, y1) >= 0.))



def main():
    cmd = " ".join(sys.argv)
    options, args = parser.parse_args(sys.argv[1:])
    conn = ea.connect(section='dessci')
    cursor = conn.cursor() # create a cursor object to handle db
    
    fname = options.outfile
    print('options:', options)
    print('outfile:', options.outfile)
    if fname is None:
        print("must specify output file.")
        exit()
    out = open(fname, 'w+')
    out.write(master_head_template.format(cmd))

    if options.zpmin:
        # Require minimum zeropoint for highly extincted measurements
        q_zp = " and oi.chip_zero_point > {0:f}".format(options.zpmin)
    else:
        q_zp = " "

    expnumfile=open(options.exptable,'r')
    expnumstr = '('
    while (1==1):
        try:
            expnumstr = expnumstr + expnumfile.readline().split()[0]
        except:
            break
        expnumstr = expnumstr + ','
    expnumstr =expnumstr[:-1] + ')' # [:-1] to throw out unneeded ','

    query = """\
    SELECT DISTINCT
        oi.ccdnum,
        oi.nite,
        oi.band,
        oi.expnum,
        (oi.chip_ccdgaina + oi.chip_ccdgainb)/2. as gain,
        oi.chip_sigsky_search,
        oi.chip_sigsky_template,
        oi.psf_nea,
        oi.chip_zero_point,
        oi.chip_zero_point_rms,
        fi.field,
        f.mjd_obs,
        fi.rac1,
        fi.decc1,
        fi.rac2,
        fi.decc2,
        fi.rac3,
        fi.decc3,
        fi.rac4,
        fi.decc4
    FROM
        marcelle.snobsinfo oi, marcelle.snforce_image fi, marcelle.snforce f
    WHERE
        oi.image_name_diff = fi.image_name_diff and 
        fi.image_name_diff = f.image_name_diff and
        oi.tile = 0 {0} and
        oi.expnum in {1}
    ORDER BY
        field, oi.ccdnum, oi.nite, oi.band
    """.format(q_zp,expnumstr)
    print(expnumstr)
    print(query)
    # Run the query or load pre-saved results.
    if os.path.exists(cache_filename):
        print("Loading from {0}...".format(cache_filename))
        data = np.load(cache_filename)
    else:
        print("Querying database (this may take a minute or two)...")
        t0 = time.time()
        QQ = cursor.execute(query) # execute query
        QQ.description
        header = [item[0] for item in cursor.description]
        print(header)
        rows = cursor.fetchall()
        # figure out dtypes for our array so we can reference names:
        dtype = []
        for head in header:
            if (head in ['CCDNUM','NITE','EXPNUM']):
                dtype.append((head.lower(),int))
            elif (head in ['BAND','FIELD']):
                dtype.append((head.lower(),"U10"))   ## S10 was giving b'Field_name' and b'filter_name' so replaced it by U10 (Simran, Nov 25, 2024)
            else:
                dtype.append((head.lower(),float))
        data = np.array(rows,dtype=dtype)
        print(data['field'])
        print(data['band'])
        print("Query took {0:.2f} seconds, {1:.2f} MB array.".format(
            time.time() - t0, data.nbytes / 1.e6))
        if options.cache:
            print("Saving to {0}...".format(cache_filename))
            np.save(cache_filename, data)
    msg = "{0:d} records".format(len(data))
    print(msg)
    out.write('#{0}\n'.format(msg))

    #-------------------------------------------------------------------------
    # BEGIN HACKS

    # Hack to remove dirty data from the table:
    # This is necessary because there were some -9's written to the database.
    mask = data['psf_nea'] > 0.
    data = data[mask]
    msg = ("HACK: {0:d} records after trimming entries with PSF_NEA < 0"
           .format(len(data)))
    print(msg)
    out.write('#{0}\n'.format(msg))

    # Hack to cut crazy zeropoint RMS values, including nan
    mask = data['chip_zero_point_rms'] <= 0.5
    data = data[mask]
    msg = ("HACK: {0:d} records after trimming entries with "
           "CHIP_ZERO_POINT_RMS > 0.5".format(len(data)))
    print(msg)
    out.write('#{0}\n'.format(msg))

    # END HACKS
    # ------------------------------------------------------------------------

    # If input file is defined, read positions.
    if options.infile is not None:
        msg = "Reading positions from {0}".format(options.infile)
        print(msg)
        positions = []
        with open(options.infile, 'r') as f:
            keys = f.readline().upper().strip().split(',')
            if ('RA' not in keys) or ('DEC' not in keys):
                print("RA and DEC must appear in first line of infile")
                exit(1)
            for line in f:
                positions.append(odict(zip(keys, line.strip().split(','))))

        # Convert RA, DEC to floats
        for pos in positions:
            pos['RA'] = float(pos['RA'])
            pos['DEC'] = float(pos['DEC'])

        ntarget = len(positions)
        msg += '... {0:d} positions'.format(ntarget)
        out.write('#{0}\n'.format(msg))

    # Otherwise, report area from which positions will be selected.
    else:
        msg = "Randomly selecting position from fields:"
        print(msg)
        out.write('#{0}\n'.format(msg))
        for name in areas.keys():
            msg = ("    {0}: RA=[{1:5.1f},{2:5.1f}] Dec=[{3:5.1f},{4:5.1f}] "
                   "{5:7.4f} deg^2, weight={6:6.4f}"
                   .format(name,
                           bounds[name][0],
                           bounds[name][1],
                           bounds[name][2],
                           bounds[name][3],
                           areas[name],
                           weights[name]))
            print(msg)
            out.write('#{0}\n'.format(msg))
        ntarget = options.n

    out.write('BEGIN LIBGEN\n')
    ntot = 0
    ngood = 0
    noverlap = 0  # keep track of number in overlapping fields
    while ngood < ntarget:
        ntot += 1

        # Get a position
        if options.infile:
            if len(positions) == 0: break
            extrakeys = positions.pop(0)
            ra = extrakeys.pop('RA')
            dec = extrakeys.pop('DEC')
        else:
            ra, dec = get_random_position()
            extrakeys = {}

        mask = point_within_quad(ra, dec,
                                 data['rac1'], data['decc1'],
                                 data['rac2'], data['decc2'],
                                 data['rac3'], data['decc3'],
                                 data['rac4'], data['decc4'])
        posdata = data[mask]

        # Only write this out if there are 2 or more images (1 image is not
        # enough to make a subtraction).
        if len(posdata) > 1:
            ngood += 1

            # Write header and entries for this position to SIMLIB file
            out.write(head_template.format(ngood, ra, dec, len(posdata),
                                           0., 0.27))

            # Write extra keys
            for key, val in extrakeys.items():
                out.write('{}: {}\n'.format(key, val))

            # Get unique fields
            fields = np.unique(posdata['field'])
            if len(fields) > 1:
                noverlap += 1


            for i, field in enumerate(fields):
                fielddata = posdata[posdata['field'] == field]
                ccds = np.unique(fielddata['ccdnum'])

                # For this field, find the first (arbitrary) image in each band 
                # and get its CHIP_ZERO_POINT and CHIP_SKYSIG_TEMPLATE
                # to be recorded as TEMPLATE_ZPT and TEMPLATE_SKYSIG.
                #
                # This is made harder by the fact that some fields don't have
                # images in every band. So there's not always a "first" band.
                # 
                # NOTE:
                # This is non-robust acrobatics!
                # This info should just be two more columns in the table.
                # Can YOU decipher what this is doing? I sure can't.
                idx = []
                for band in ['g', 'r', 'i', 'z']:
                    try:
                        idx.append(np.flatnonzero(fielddata['band']==band)[0])
                    except:
                        idx.append(-1)  # -1 indicates no image in given band
                zpstr = ["%6.3f" % fielddata['chip_zero_point'][k]
                         if k != -1 else "0.0" for k in idx]
                zpstr = " ".join(zpstr)
                skystr = ["%7.2f" % fielddata['chip_sigsky_template'][k]
                          if k != -1 else "0.0" for k in idx]
                skystr = " ".join(skystr)

                out.write(field_template.format(field, str(ccds),
                                                zpstr, skystr))
                if i == 0:
                    out.write(head_comment)

                for j in range(len(fielddata)):
                    out.write(
                        line_template.format(
                            'S',
                            float(fielddata['mjd_obs'][j]),
                            int(fielddata['expnum'][j]),
                            str(fielddata['band'][j]),
                            float(fielddata['gain'][j]),
                            0.,
                            float(fielddata['chip_sigsky_search'][j]),
                            float(fielddata['psf_nea'][j]),
                            0.,
                            0.,
                            float(fielddata['chip_zero_point'][j]),
                            float(fielddata['chip_zero_point_rms'][j]),
                            99.)
                        )

            out.write('END_LIBID: {0:d}\n'.format(ngood))

        print('\r{0:<5d} on images / {1:<5d} placed'.format(ngood, ntot), end=" ") 
        sys.stdout.flush()

    out.write("END_OF_SIMLIB:\n")

    print("\n{0:<5d} on overlapping fields".format(noverlap))

    # Effective area. Only calculated if positions were seleceted randomly.
    if options.infile is None:
        effective_area = totarea * float(ngood)/ntot
        print("Effective area: {0:.4f} deg^2".format(effective_area))
        out.write('EFFECTIVE_AREA: {0:.4f}\n'.format(effective_area))

    out.close()
    print("Wrote to:", fname)

if __name__ == '__main__':
    main()

 
