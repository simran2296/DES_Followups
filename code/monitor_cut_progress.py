# A module to monitor background run_single_cuts.py jobs

import os
import sys
import time

event_name = sys.argv[1]
sim_include = sys.argv[2:]

log_dir = '../events/%s/logs/' %event_name

running_dict = {k: True for k in sim_include}
phase_dict = {k: 1 for k in sim_include}
progress_dict = {k: "Initializing" for k in sim_include}

#phases:
# 1: initializing
# 2: processing
# 3: cutting
# 4: getting cut results (analyzing)
# 5: finalizing
# 6: done

running = eval(' or '.join([str(x) for x in running_dict.values()]))
while running:
    
    #add fancier progress tracking stuff here
    for obj in sim_include:
        if phase_dict[obj] <= 2 and os.path.exists(log_dir + 'parse_%s.log' %obj):
            phase_dict[obj] = 2

            try:
                stream = open(log_dir + 'parse_%s.log' %obj, 'r')
                progress = stream.readlines()[0]
                stream.close()
            except:
                #print("WARNING: %s parse log file was never created. Something may have broken previously" %obj)
                progress_dict[obj] = 'Initializing'
                #phase_dict[obj] = 7 #no conditions will be met, so the obj will be skipped
                continue
            

            if int(float(progress)) == 100:
                phase_dict[obj] = 3

            progress_dict[obj] = progress + ' %'

        elif phase_dict[obj] == 3:
            progress_dict[obj] = 'Cutting'

            if os.path.exists(log_dir + 'cut_%s.DONE' %obj):
                phase_dict[obj] = 4
                progress_dict[obj] = 'Analyzing'

        elif phase_dict[obj] == 4:
            progress_dict[obj] = 'Analyzing' #technically already set

            if os.path.exists(log_dir + 'get_results_%s.DONE' %obj):
                phase_dict[obj] = 5
        
        elif phase_dict[obj] == 5:
            #next time through, everything should be wrapping up
            progress_dict[obj] = 'Finalizing'
            phase_dict[obj] = 6
        elif phase_dict[obj] == 6:
            progress_dict[obj] = 'Done!'

    #write progress to screen
    string_output = ['%s -- %s' %(k, v) for k, v in progress_dict.items()]
    progress_out = 'Processing:  ' + ' | '.join(string_output) + ' |' + ' ' * 30
    sys.stdout.write('\r' + progress_out)
    sys.stdout.flush()

    #exit conditions
    for obj in sim_include:
        if os.path.exists(log_dir + 'run_cuts_%s.DONE' %obj):
            running_dict[obj] = False

    running = eval(' or '.join([str(x) for x in running_dict.values()]))

    time.sleep(2)


#final done statement
sys.stdout.write('\rProcessing:  ' + ' | '.join([k + ' -- Done!' for k in sim_include]) + ' |' + ' ' * 30 + '\n')
