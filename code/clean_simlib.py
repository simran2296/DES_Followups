# A code to remove duplicate rows in SIMLIB file to prevent erroneous SNANA sims

import sys
import numpy as np

# Target SIMLIB file
filename = str(sys.argv[1])
outfile_name = str(sys.argv[2])
area = str(sys.argv[3])
f = open(filename, 'r')
lines = f.readlines()
f.close()

if len(sys.argv) > 4:
    nobs_force = int(sys.argv[4])
    trim_libids_by_nobs = True
else:
    trim_libids_by_nobs = False

# Collect indices of each LIBID start and end, and the line with NOBS
libid_starts, libid_ends, nobs_lines = [], [], []
index = 0
for line in lines:
    if line[0:6] == 'LIBID:': libid_starts.append(index)
    if line[0:10] == 'END_LIBID:': libid_ends.append(index)
    if line[0:4] == 'RA: ': nobs_lines.append(index)
    index += 1

# Clean the LIBIDs incrementally
libid_num = 1
lines_to_delete = []
for libid in libid_starts:
    line_num = libid + 0
    expnums = []
    while line_num <= libid_ends[libid_num - 1]:
        line_values = lines[line_num].split(' ')
        line_values = [x for x in line_values if x != '']
        if line_values[0] == 'S:':
            expnum = int(line_values[2])
            expnums.append(expnum)
        line_num += 1
    expnum_array = np.asarray(expnums)
    values, counts = np.unique(expnum_array, return_counts=True)
    duplicated_expnums = values[counts > 1]
    num_duplicated_expnums = len(duplicated_expnums)

    if num_duplicated_expnums > 0:
        #cycle through libid to find the line numbers with the same expnums
        line_num = libid + 0
        used_expnums = []
        while line_num <= libid_ends[libid_num - 1]:
            line_values = lines[line_num].split(' ')
            line_values = [x for x in line_values if x != '']
            if line_values[0] == 'S:':
                expnum = int(line_values[2])
                if expnum in duplicated_expnums and expnum not in used_expnums:
                    lines_to_delete.append(line_num)
                    used_expnums.append(expnum)
                    
            line_num += 1

        #Lines to delete has been stored, just need to update n_obs
        n_obs_line = nobs_lines[libid_num - 1]
        n_obs_position_in_line = lines[n_obs_line].find('NOBS:')
        n_obs = int(lines[n_obs_line][n_obs_position_in_line + 6:-1])
        new_n_obs = n_obs - num_duplicated_expnums
        replacement_line = lines[n_obs_line][0:n_obs_position_in_line + 6]
        replacement_line += str(new_n_obs)
        replacement_line += '\n'
        lines[n_obs_line] = replacement_line
              
    libid_num += 1
    
# Delete all lines marked as duplicate
for ii in sorted(lines_to_delete, reverse=True):
    del lines[ii]

# Overwrite field lines in simlib to be an int 0-9
outlines = []
for line in lines:
    if line[0:6] == 'FIELD:':
        outlines.append('FIELD: 1\n')
    else:
        outlines.append(line)

#overwrite the effective area line
outlines[-1] = 'EFFECTIVE_AREA: %s' %area

# if desired by user, drop libids that have fewer than specified number of observations

if trim_libids_by_nobs:
    new_outlines = []
    libids_to_delete = []
    new_line_counter = 0
    while new_line_counter < len(outlines) - 1:
        if outlines[new_line_counter + 1].find('NOBS') != -1:
            nobs = int(outlines[new_line_counter + 1].strip().split(' ')[-1])
            if nobs < nobs_force:
                #mark libid for deletion
                libids_to_delete.append(outlines[new_line_counter].strip().split(' ')[-1])
            new_line_counter += nobs
        else:
            new_line_counter += 1

    bad_libid_starts = []
    bad_libid_ends = []
    for line_index in range(len(outlines)):
        if outlines[line_index][0:6] == 'LIBID:':
            libid_index = outlines[line_index].strip().split(' ')[-1]
            if libid_index in libids_to_delete:
                bad_libid_starts.append(line_index)
        elif outlines[line_index][0:10] == 'END_LIBID:':
            libid_index= outlines[line_index].strip().split(' ')[-1]
            if libid_index in libids_to_delete:
                bad_libid_ends.append(line_index)
    
    bad_libid_indices = []
    for x, y in zip(bad_libid_starts, bad_libid_ends):
        l = range(x - 2, y + 1)
        for item in l:
            bad_libid_indices.append(item)

    for ii in range(len(outlines)):
        if ii not in bad_libid_indices:
            new_outlines.append(outlines[ii])

    outlines = new_outlines


# Write output SIMLIB file
outfile = open(outfile_name, 'w+')
for line in outlines:
    print(line, file=outfile)
    #print >>outfile, line, #this was python2
outfile.close()
