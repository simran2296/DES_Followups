# A module to show progress of simulations running in the background

import os
import sys
import time

#locate log files
event_name = sys.argv[1]
agn_log_file = '../events/%s/logs/sim_agn.log' %event_name
kn_log_file = '../events/%s/logs/sim_kn.log' %event_name
ia_log_file = '../events/%s/logs/sim_ia.log' %event_name
cc_log_file = '../events/%s/logs/sim_cc.log' %event_name
cart_log_file = '../events/%s/logs/sim_cart.log' %event_name
ilot_log_file = '../events/%s/logs/sim_ilot.log' %event_name
mdwarf_log_file = '../events/%s/logs/sim_mdwarf.log' %event_name
sn91bg_log_file = '../events/%s/logs/sim_91bg.log' %event_name
iax_log_file = '../events/%s/logs/sim_iax.log' %event_name
pia_log_file = '../events/%s/logs/sim_pia.log' %event_name
slsn_log_file = '../events/%s/logs/sim_slsn.log' %event_name
tde_log_file = '../events/%s/logs/sim_tde.log' %event_name
bbh_log_file = '../events/%s/logs/sim_bbh.log' %event_name
agn_tr_log_file = '../events/%s/logs/sim_tr_agn.log' %event_name
kn_tr_log_file = '../events/%s/logs/sim_tr_kn.log' %event_name
ia_tr_log_file = '../events/%s/logs/sim_tr_ia.log' %event_name
cc_tr_log_file = '../events/%s/logs/sim_tr_cc.log' %event_name
cart_tr_log_file = '../events/%s/logs/sim_tr_cart.log' %event_name
ilot_tr_log_file = '../events/%s/logs/sim_tr_ilot.log' %event_name
mdwarf_tr_log_file = '../events/%s/logs/sim_tr_mdwarf.log' %event_name
sn91bg_tr_log_file = '../events/%s/logs/sim_tr_91bg.log' %event_name
iax_tr_log_file = '../events/%s/logs/sim_tr_iax.log' %event_name
pia_tr_log_file = '../events/%s/logs/sim_tr_pia.log' %event_name
slsn_tr_log_file = '../events/%s/logs/sim_tr_slsn.log' %event_name
tde_tr_log_file = '../events/%s/logs/sim_tr_tde.log' %event_name
bbh_tr_log_file = '../events/%s/logs/sim_tr_bbh.log' %event_name

#determine which sims are being run
agn_running = True if 'AGN' in sys.argv else False
ia_running = True if 'Ia' in sys.argv else False
cc_running = True if 'CC' in sys.argv else False
kn_running = True if 'KN' in sys.argv else False
cart_running = True if 'CaRT' in sys.argv else False
ilot_running = True if 'ILOT' in sys.argv else False
mdwarf_running = True if 'Mdwarf' in sys.argv else False
sn91bg_running = True if 'SN91bg' in sys.argv else False
iax_running = True if 'Iax' in sys.argv else False
pia_running = True if 'PIa' in sys.argv else False
slsn_running = True if 'SLSN' in sys.argv else False
tde_running = True if 'TDE' in sys.argv else False
bbh_running = True if 'BBH' in sys.argv else False
agn_tr_running = True if 'AGN-tr' in sys.argv else False
ia_tr_running = True if 'Ia-tr' in sys.argv else False
cc_tr_running = True if 'CC-tr' in sys.argv else False
kn_tr_running = True if 'KN-tr' in sys.argv else False
cart_tr_running = True if 'CaRT-tr' in sys.argv else False
ilot_tr_running = True if 'ILOT-tr' in sys.argv else False
mdwarf_tr_running = True if 'Mdwarf-tr' in sys.argv else False
sn91bg_tr_running = True if 'SN91bg-tr' in sys.argv else False
iax_tr_running = True if 'Iax-tr' in sys.argv else False
pia_tr_running = True if 'PIa-tr' in sys.argv else False
slsn_tr_running = True if 'SLSN-tr' in sys.argv else False
tde_tr_running = True if 'TDE-tr' in sys.argv else False
bbh_tr_running = True if 'BBH-tr' in sys.argv else False


#track errors
agn_error = False
ia_error = False
cc_error = False
kn_error = False
cart_error = False
ilot_error = False
mdwarf_error = False
sn91bg_error = False
iax_error = False
pia_error = False
slsn_error = False
tde_error = False
bbh_error = False
agn_tr_error = False
ia_tr_error = False
cc_tr_error = False
kn_tr_error = False
cart_tr_error = False
ilot_tr_error = False
mdwarf_tr_error = False
sn91bg_tr_error = False
iax_tr_error = False
pia_tr_error = False
slsn_tr_error = False
tde_tr_error = False
bbh_tr_error = False

#start monitoring
running = agn_running or kn_running or ia_running or cc_running or cart_running or ilot_running or mdwarf_running or sn91bg_running or iax_running or pia_running or slsn_running or tde_running or bbh_running
running = running or agn_tr_running or kn_tr_running or ia_tr_running or cc_tr_running or cart_tr_running or ilot_tr_running or mdwarf_tr_running or sn91bg_tr_running or iax_tr_running or iax_tr_running or pia_tr_running or slsn_tr_running or tde_tr_running or bbh_tr_running
agn_phase1, agn_phase2, agn_phase3 = True, False, False
kn_phase1, kn_phase2, kn_phase3, kn_phase4 = True, False, False, False
ia_phase1, ia_phase2, ia_phase3 = True, False, False
cc_phase1, cc_phase2, cc_phase3 = True, False, False
cart_phase1, cart_phase2, cart_phase3, cart_phase4 = True, False, False, False
ilot_phase1, ilot_phase2, ilot_phase3, ilot_phase4 = True, False, False, False
mdwarf_phase1, mdwarf_phase2, mdwarf_phase3 = True, False, False
sn91bg_phase1, sn91bg_phase2, sn91bg_phase3, sn91bg_phase4 = True, False, False, False
iax_phase1, iax_phase2, iax_phase3, iax_phase4 = True, False, False, False
pia_phase1, pia_phase2, pia_phase3, pia_phase4 = True, False, False, False
slsn_phase1, slsn_phase2, slsn_phase3, slsn_phase4 = True, False, False, False
tde_phase1, tde_phase2, tde_phase3, tde_phase4 = True, False, False, False
bbh_phase1, bbh_phase2, bbh_phase3, bbh_phase4 = True, False, False, False

agn_tr_phase1, agn_tr_phase2, agn_tr_phase3 = True, False, False
kn_tr_phase1, kn_tr_phase2, kn_tr_phase3, kn_tr_phase4 = True, False, False, False
ia_tr_phase1, ia_tr_phase2, ia_tr_phase3 = True, False, False
cc_tr_phase1, cc_tr_phase2, cc_tr_phase3 = True, False, False
cart_tr_phase1, cart_tr_phase2, cart_tr_phase3, cart_tr_phase4 = True, False, False, False
ilot_tr_phase1, ilot_tr_phase2, ilot_tr_phase3, ilot_tr_phase4 = True, False, False, False
mdwarf_tr_phase1, mdwarf_tr_phase2, mdwarf_tr_phase3 = True, False, False
sn91bg_tr_phase1, sn91bg_tr_phase2, sn91bg_tr_phase3, sn91bg_tr_phase4 = True, False, False, False
iax_tr_phase1, iax_tr_phase2, iax_tr_phase3, iax_tr_phase4 = True, False, False, False
pia_tr_phase1, pia_tr_phase2, pia_tr_phase3, pia_tr_phase4 = True, False, False, False
slsn_tr_phase1, slsn_tr_phase2, slsn_tr_phase3, slsn_tr_phase4 = True, False, False, False
tde_tr_phase1, tde_tr_phase2, tde_tr_phase3, tde_tr_phase4 = True, False, False, False
bbh_tr_phase1, bbh_tr_phase2, bbh_tr_phase3, bbh_tr_phase4 = True, False, False, False

#let the sims get started
time.sleep(6)

while running:

    """
    AGN Progress Tracking
    """
    if agn_running:
        agn_log = open(agn_log_file, 'r')
        agn_info = agn_log.readlines()

        #check for errors and exit the loop if necessary
        for line in agn_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_agn.log")
                agn_error = True
                agn_running = False

        last_lines = agn_info[-30:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]

        agn_log.close()

        if agn_phase3 or (not agn_phase1 and last_line.find('Finished writing') == -1):
            agn_phase1 = False
            agn_phase2 = False
            agn_phase3 = True
            agn_running = False ## it should finish up in time where this is okay
            agn_progress = 'Finishing'
        elif agn_phase2 or last_line.find('Finished writing') != -1:
            agn_phase1 = False
            agn_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            agn_progress = '%.2f %%' % (float(completed) / float(total) * 100.0)
        elif agn_phase1 == True:
            agn_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i" %(int(agn_phase1), int(agn_phase2), int(agn_phase3)))
            print("\tLast line: %s" %last_line)
            agn_running = False
    else:
        agn_progress = 'Done!'
        if agn_error: agn_progress = 'ERROR'
            

    """
    AGN-tr Progress Tracking
    """
    if agn_tr_running:
        agn_tr_log = open(agn_tr_log_file, 'r')
        agn_tr_info = agn_tr_log.readlines()

        #check for errors and exit the loop if necessary
        for line in agn_tr_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_tr_agn.log")
                agn_tr_error = True
                agn_tr_running = False

        last_lines = agn_tr_info[-30:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]

        agn_tr_log.close()

        if agn_tr_phase3 or (not agn_tr_phase1 and last_line.find('Finished writing') == -1):
            agn_tr_phase1 = False
            agn_tr_phase2 = False
            agn_tr_phase3 = True
            agn_tr_running = False ## it should finish up in time where this is okay
            agn_tr_progress = 'Finishing'
        elif agn_tr_phase2 or last_line.find('Finished writing') != -1:
            agn_tr_phase1 = False
            agn_tr_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            agn_tr_progress = '%.2f %%' % (float(completed) / float(total) * 100.0)
        elif agn_tr_phase1 == True:
            agn_tr_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i" %(int(agn_tr_phase1), int(agn_tr_phase2), int(agn_tr_phase3)))
            print("\tLast line: %s" %last_line)
            agn_tr_running = False
    else:
        agn_tr_progress = 'Done!'
        if agn_tr_error: agn_tr_progress = 'ERROR'

            
    """
    Ia Progress Tracking
    """
    if ia_running:
        ia_log = open(ia_log_file, 'r')
        ia_info = ia_log.readlines()

        #check for errors and exit the loop if necessary
        for line in ia_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_ia.log")
                ia_running = False
                ia_error = True

        last_line = ia_info[-1]
        ia_log.close()

        if ia_phase3 or (not ia_phase1 and last_line.find('Finished writing') == -1):
            ia_phase1 = False
            ia_phase2 = False
            ia_phase3 = True
            ia_running = False ## it should finish up in time where this is okay
            ia_progress = 'Finishing'
        elif ia_phase2 or last_line.find('Finished writing') != -1:
            #print("Now tracking progress")
            ia_phase1 = False
            ia_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            ia_progress = '%.2f %%' % (float(completed) / float(total) * 100.0)
        elif ia_phase1 == True:
            ia_progress = 'Initializing'
            #print("In phase 1")
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i" %(int(ia_phase1), int(ia_phase2), int(ia_phase3)))
            print("\tLast line: %s" %last_line)
            ia_running = False
    else:
        ia_progress = 'Done!'
        if ia_error: ia_progress = 'ERROR'

    """
    Ia-tr Progress Tracking
    """
    if ia_tr_running:
        ia_tr_log = open(ia_tr_log_file, 'r')
        ia_tr_info = ia_tr_log.readlines()

        #check for errors and exit the loop if necessary
        for line in ia_tr_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_tr_ia.log")
                ia_tr_running = False
                ia_error = True

        last_line = ia_tr_info[-1]
        ia_tr_log.close()

        if ia_tr_phase3 or (not ia_tr_phase1 and last_line.find('Finished writing') == -1):
            ia_tr_phase1 = False
            ia_tr_phase2 = False
            ia_tr_phase3 = True
            ia_tr_running = False ## it should finish up in time where this is okay
            ia_tr_progress = 'Finishing'
        elif ia_tr_phase2 or last_line.find('Finished writing') != -1:
            ia_tr_phase1 = False
            ia_tr_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            ia_tr_progress = '%.2f %%' % (float(completed) / float(total) * 100.0)
        elif ia_tr_phase1 == True:
            ia_tr_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i" %(int(ia_tr_phase1), int(ia_tr_phase2), int(ia_tr_phase3)))
            print("\tLast line: %s" %last_line)
            ia_tr_running = False
    else:
        ia_tr_progress = 'Done!'
        if ia_tr_error: ia_tr_progress = 'ERROR'

    """
    CC Progress Tracking
    """
    if cc_running:
        cc_log = open(cc_log_file, 'r')
        cc_info = cc_log.readlines()

        #check for errors and exit the loop if necessary
        for line in cc_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_cc.log")
                cc_running = False
                cc_error = True

        ####last_line must pick the final 'Finished generating' line
        #last_lines = cc_info[-70:]
        last_lines = cc_info[-100:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        cc_log.close()

        if cc_phase3 or (not cc_phase1 and last_line.find('Finished writing') == -1):
            cc_phase1 = False
            cc_phase2 = False
            cc_phase3 = True
            cc_running = False ## it should finish up in time where this is okay
            cc_progress = 'Finishing'
        elif cc_phase2 or last_line.find('Finished writing') != -1:
            cc_phase1 = False
            cc_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            cc_progress = '%.2f %%' % (float(completed) / float(total) * 100.0)
        elif cc_phase1 == True:
            cc_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i" %(int(cc_phase1), int(cc_phase2), int(cc_phase3)))
            print("\tLast line: %s" %last_line)
            cc_running = False
    else:
        cc_progress = 'Done!'
        if cc_error: cc_progress = 'ERROR'
    
    """
    CC-tr Progress Tracking
    """
    if cc_tr_running:
        cc_tr_log = open(cc_tr_log_file, 'r')
        cc_tr_info = cc_tr_log.readlines()

        #check for errors and exit the loop if necessary
        for line in cc_tr_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_tr_cc.log")
                cc_tr_running = False
                cc_tr_error = True

        ####last_line must pick the final 'Finished generating' line
        last_lines = cc_tr_info[-70:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        cc_tr_log.close()

        if cc_tr_phase3 or (not cc_tr_phase1 and last_line.find('Finished writing') == -1):
            cc_tr_phase1 = False
            cc_tr_phase2 = False
            cc_tr_phase3 = True
            cc_tr_running = False ## it should finish up in time where this is okay
            cc_tr_progress = 'Finishing'
        elif cc_tr_phase2 or last_line.find('Finished writing') != -1:
            cc_tr_phase1 = False
            cc_tr_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            cc_tr_progress = '%.2f %%' % (float(completed) / float(total) * 100.0)
        elif cc_tr_phase1 == True:
            cc_tr_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i" %(int(cc_tr_phase1), int(cc_tr_phase2), int(cc_tr_phase3)))
            print("\tLast line: %s" %last_line)
            cc_tr_running = False
    else:
        cc_tr_progress = 'Done!'
        if cc_tr_error: cc_tr_progress = 'ERROR'
            
    """
    KN Progress Tracking
    """
    if kn_running:
        kn_log = open(kn_log_file, 'r')
        kn_info = kn_log.readlines()

        #check for errors and exit the loop if necessary
        for line in kn_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_kn.log")
                kn_running = False
                kn_error = True

        last_lines = kn_info[-30:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1 or line.find('Read  (ised=') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        
        #last_line = kn_info[-2]
        kn_log.close()
        
        if kn_phase4 or (not kn_phase1 and not kn_phase2 and last_line.find('Finished writing') == -1):
            kn_phase1 = False
            kn_phase2 = False
            kn_phase3 = False
            kn_phase4 = True
            kn_running = False ## it should finish up in time where this is okay
            kn_progress = 'Finishing'
        elif kn_phase3 or last_line.find('Finished writing') != -1:
            kn_phase1 = False
            kn_phase2 = False
            kn_phase3 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            kn_progress = '%.2f %%' % (float(completed) / float(total) * 20.0 + 80.0)
        elif kn_phase2 or last_line.find('Read  (ised=') != -1:
            kn_phase1 = False
            kn_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n'][1]
            more_info = info.split('=')[-1].split(')')[0].split('/')
            total = more_info[1]
            completed = more_info[0]
            if total == 'with':
                kn_progress = 'Done!'
                kn_running = False
            else:
                kn_progress = '%.2f %%' % (float(completed) / float(total) * 80.0)
        elif kn_phase1 == True:
            kn_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i, Phase4: %i" %(int(kn_phase1), int(kn_phase2), int(kn_phase3), int(kn_phase4)))
            print("\tLast line: %s" %last_line)
            kn_running = False
    else:
        kn_progress = 'Done!'
        if kn_error: kn_progress = 'ERROR'

    """
    KN-tr Progress Tracking
    """
    if kn_tr_running:
        kn_tr_log = open(kn_tr_log_file, 'r')
        kn_tr_info = kn_tr_log.readlines()

        #check for errors and exit the loop if necessary
        for line in kn_tr_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_tr_kn.log")
                kn_tr_running = False
                kn_tr_error = True

        last_lines = kn_tr_info[-30:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1 or line.find('Read  (ised=') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        
        #last_line = kn_info[-2]
        kn_tr_log.close()
        
        if kn_tr_phase4 or (not kn_tr_phase1 and not kn_tr_phase2 and last_line.find('Finished writing') == -1):
            kn_tr_phase1 = False
            kn_tr_phase2 = False
            kn_tr_phase3 = False
            kn_tr_phase4 = True
            kn_tr_running = False ## it should finish up in time where this is okay
            kn_tr_progress = 'Finishing'
        elif kn_tr_phase3 or last_line.find('Finished writing') != -1:
            kn_tr_phase1 = False
            kn_tr_phase2 = False
            kn_tr_phase3 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            kn_tr_progress = '%.2f %%' % (float(completed) / float(total) * 20.0 + 80.0)
        elif kn_tr_phase2 or last_line.find('Read  (ised=') != -1:
            kn_tr_phase1 = False
            kn_tr_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n'][1]
            more_info = info.split('=')[-1].split(')')[0].split('/')
            total = more_info[1]
            completed = more_info[0]
            if total == 'with':
                kn_tr_progress = 'Done!'
                kn_tr_running = False
            else:
                kn_tr_progress = '%.2f %%' % (float(completed) / float(total) * 80.0)
        elif kn_tr_phase1 == True:
            kn_tr_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i, Phase4: %i" %(int(kn_tr_phase1), int(kn_tr_phase2), int(kn_tr_phase3), int(kn_tr_phase4)))
            print("\tLast line: %s" %last_line)
            kn_tr_running = False
    else:
        kn_tr_progress = 'Done!'
        if kn_tr_error: kn_tr_progress = 'ERROR'

    """
    CaRT Progress Tracking
    """
    if cart_running:
        cart_log = open(cart_log_file, 'r')
        cart_info = cart_log.readlines()

        #check for errors and exit the loop if necessary
        for line in cart_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_cart.log")
                cart_running = False
                cart_error = True

        last_lines = cart_info[-40:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1 or line.find('Read  (ised=') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        
        cart_log.close()
        
        if cart_phase4 or (not cart_phase1 and not cart_phase2 and last_line.find('Finished writing') == -1):
            cart_phase1 = False
            cart_phase2 = False
            cart_phase3 = False
            cart_phase4 = True
            cart_running = False ## it should finish up in time where this is okay
            cart_progress = 'Finishing'
        elif cart_phase3 or last_line.find('Finished writing') != -1:
            cart_phase1 = False
            cart_phase2 = False
            cart_phase3 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            cart_progress = '%.2f %%' % (float(completed) / float(total) * 20.0 + 80.0)
        elif cart_phase2 or last_line.find('Read  (ised=') != -1:
            cart_phase1 = False
            cart_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n'][1]
            more_info = info.split('=')[-1].split(')')[0].split('/')
            try:
                total = more_info[1]
            except:
                print('\n' + last_line + '\n')
                total = 700
            completed = more_info[0]
            cart_progress = '%.2f %%' % (float(completed) / float(total) * 80.0)
        elif cart_phase1 == True:
            cart_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i, Phase4: %i" %(int(cart_phase1), int(cart_phase2), int(cart_phase3), int(cart_phase4)))
            print("\tLast line: %s" %last_line)
            cart_running = False
    else:
        cart_progress = 'Done!'
        if cart_error: cart_progress = 'ERROR'

    """
    CaRT-tr Progress Tracking
    """
    if cart_tr_running:
        cart_tr_log = open(cart_tr_log_file, 'r')
        cart_tr_info = cart_tr_log.readlines()

        #check for errors and exit the loop if necessary
        for line in cart_tr_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_tr_cart.log")
                cart_tr_running = False
                cart_tr_error = True

        last_lines = cart_tr_info[-40:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1 or line.find('Read  (ised=') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        
        cart_tr_log.close()
        
        if cart_tr_phase4 or (not cart_tr_phase1 and not cart_tr_phase2 and last_line.find('Finished writing') == -1):
            cart_tr_phase1 = False
            cart_tr_phase2 = False
            cart_tr_phase3 = False
            cart_tr_phase4 = True
            cart_tr_running = False ## it should finish up in time where this is okay
            cart_tr_progress = 'Finishing'
        elif cart_tr_phase3 or last_line.find('Finished writing') != -1:
            cart_tr_phase1 = False
            cart_tr_phase2 = False
            cart_tr_phase3 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            cart_tr_progress = '%.2f %%' % (float(completed) / float(total) * 20.0 + 80.0)
        elif cart_tr_phase2 or last_line.find('Read  (ised=') != -1:
            cart_tr_phase1 = False
            cart_tr_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n'][1]
            more_info = info.split('=')[-1].split(')')[0].split('/')
            try:
                total = more_info[1]
            except:
                print('\n' + last_line + '\n')
                total = 700
            completed = more_info[0]
            cart_tr_progress = '%.2f %%' % (float(completed) / float(total) * 80.0)
        elif cart_tr_phase1 == True:
            cart_tr_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i, Phase4: %i" %(int(cart_tr_phase1), int(cart_tr_phase2), int(cart_tr_phase3), int(cart_tr_phase4)))
            print("\tLast line: %s" %last_line)
            cart_tr_running = False
    else:
        cart_tr_progress = 'Done!'
        if cart_tr_error: cart_tr_progress = 'ERROR'

    """
    ILOT Progress Tracking
    """
    if ilot_running:
        ilot_log = open(ilot_log_file, 'r')
        ilot_info = ilot_log.readlines()

        #check for errors and exit the loop if necessary
        for line in ilot_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_ilot.log")
                ilot_running = False
                ilot_error = True

        last_lines = ilot_info[-10:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1 or line.find('Read  (ised=') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        
        ilot_log.close()
        
        if ilot_phase4 or (not ilot_phase1 and not ilot_phase2 and last_line.find('Finished writing') == -1):
            ilot_phase1 = False
            ilot_phase2 = False
            ilot_phase3 = False
            ilot_phase4 = True
            ilot_running = False ## it should finish up in time where this is okay
            ilot_progress = 'Finishing'
        elif ilot_phase3 or last_line.find('Finished writing') != -1:
            ilot_phase1 = False
            ilot_phase2 = False
            ilot_phase3 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            ilot_progress = '%.2f %%' % (float(completed) / float(total) * 20.0 + 80.0)
        elif ilot_phase2 or last_line.find('Read  (ised=') != -1:
            ilot_phase1 = False
            ilot_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n'][1]
            more_info = info.split('=')[-1].split(')')[0].split('/')
            total = more_info[1]
            completed = more_info[0]
            ilot_progress = '%.2f %%' % (float(completed) / float(total) * 80.0)
        elif ilot_phase1 == True:
            ilot_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i, Phase4: %i" %(int(ilot_phase1), int(ilot_phase2), int(ilot_phase3), int(ilot_phase4)))
            print("\tLast line: %s" %last_line)
            ilot_running = False
    else:
        ilot_progress = 'Done!'
        if ilot_error: ilot_progress = 'ERROR'

    """
    ILOT-tr Progress Tracking
    """
    if ilot_tr_running:
        ilot_tr_log = open(ilot_tr_log_file, 'r')
        ilot_tr_info = ilot_tr_log.readlines()

        #check for errors and exit the loop if necessary
        for line in ilot_tr_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_tr_ilot.log")
                ilot_tr_running = False
                ilot_tr_error = True

        last_lines = ilot_tr_info[-10:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1 or line.find('Read  (ised=') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        
        ilot_tr_log.close()
        
        if ilot_tr_phase4 or (not ilot_tr_phase1 and not ilot_tr_phase2 and last_line.find('Finished writing') == -1):
            ilot_tr_phase1 = False
            ilot_tr_phase2 = False
            ilot_tr_phase3 = False
            ilot_tr_phase4 = True
            ilot_tr_running = False ## it should finish up in time where this is okay
            ilot_tr_progress = 'Finishing'
        elif ilot_tr_phase3 or last_line.find('Finished writing') != -1:
            ilot_tr_phase1 = False
            ilot_tr_phase2 = False
            ilot_tr_phase3 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            ilot_tr_progress = '%.2f %%' % (float(completed) / float(total) * 20.0 + 80.0)
        elif ilot_tr_phase2 or last_line.find('Read  (ised=') != -1:
            ilot_tr_phase1 = False
            ilot_tr_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n'][1]
            more_info = info.split('=')[-1].split(')')[0].split('/')
            total = more_info[1]
            completed = more_info[0]
            ilot_tr_progress = '%.2f %%' % (float(completed) / float(total) * 80.0)
        elif ilot_tr_phase1 == True:
            ilot_tr_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i, Phase4: %i" %(int(ilot_tr_phase1), int(ilot_tr_phase2), int(ilot_tr_phase3), int(ilot_tr_phase4)))
            print("\tLast line: %s" %last_line)
            ilot_tr_running = False
    else:
        ilot_tr_progress = 'Done!'
        if ilot_tr_error: ilot_tr_progress = 'ERROR'

    """
    Mdwarf Progress Tracking
    """
    if mdwarf_running:
        mdwarf_log = open(mdwarf_log_file, 'r')
        mdwarf_info = mdwarf_log.readlines()

        #check for errors and exit the loop if necessary
        for line in mdwarf_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_mdwarf.log")
                mdwarf_running = False
                mdwarf_error = True

        ####last_line must pick the final 'Finished generating' line
        last_lines = mdwarf_info[-70:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        mdwarf_log.close()

        if mdwarf_phase3 or (not mdwarf_phase1 and last_line.find('Finished writing') == -1):
            mdwarf_phase1 = False
            mdwarf_phase2 = False
            mdwarf_phase3 = True
            mdwarf_running = False ## it should finish up in time where this is okay
            mdwarf_progress = 'Finishing'
        elif mdwarf_phase2 or last_line.find('Finished writing') != -1:
            mdwarf_phase1 = False
            mdwarf_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            mdwarf_progress = '%.2f %%' % (float(completed) / float(total) * 100.0)
        elif mdwarf_phase1 == True:
            mdwarf_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i" %(int(mdwarf_phase1), int(mdwarf_phase2), int(mdwarf_phase3)))
            print("\tLast line: %s" %last_line)
            mdwarf_running = False
    else:
        mdwarf_progress = 'Done!'
        if mdwarf_error: mdwarf_progress = 'ERROR'

    """
    Mdwarf-tr Progress Tracking
    """
    if mdwarf_tr_running:
        mdwarf_tr_log = open(mdwarf_tr_log_file, 'r')
        mdwarf_tr_info = mdwarf_tr_log.readlines()

        #check for errors and exit the loop if necessary
        for line in mdwarf_tr_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_tr_mdwarf.log")
                mdwarf_tr_running = False
                mdwarf_tr_error = True

        ####last_line must pick the final 'Finished generating' line
        last_lines = mdwarf_tr_info[-70:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        mdwarf_tr_log.close()

        if mdwarf_tr_phase3 or (not mdwarf_tr_phase1 and last_line.find('Finished writing') == -1):
            mdwarf_tr_phase1 = False
            mdwarf_tr_phase2 = False
            mdwarf_tr_phase3 = True
            mdwarf_tr_running = False ## it should finish up in time where this is okay
            mdwarf_tr_progress = 'Finishing'
        elif mdwarf_tr_phase2 or last_line.find('Finished writing') != -1:
            mdwarf_tr_phase1 = False
            mdwarf_tr_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            mdwarf_tr_progress = '%.2f %%' % (float(completed) / float(total) * 100.0)
        elif mdwarf_tr_phase1 == True:
            mdwarf_tr_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i" %(int(mdwarf_tr_phase1), int(mdwarf_tr_phase2), int(mdwarf_tr_phase3)))
            print("\tLast line: %s" %last_line)
            mdwarf_tr_running = False
    else:
        mdwarf_tr_progress = 'Done!'
        if mdwarf_tr_error: mdwarf_tr_progress = 'ERROR'

    """
    SNIa91bg Progress Tracking
    """
    if sn91bg_running:
        sn91bg_log = open(sn91bg_log_file, 'r')
        sn91bg_info = sn91bg_log.readlines()

        #check for errors and exit the loop if necessary
        for line in sn91bg_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_91bg.log")
                sn91bg_running = False
                sn91bg_error = True

        last_lines = sn91bg_info[-10:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1 or line.find('Read  (ised=') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        
        sn91bg_log.close()
        
        if sn91bg_phase4 or (not sn91bg_phase1 and not sn91bg_phase2 and last_line.find('Finished writing') == -1):
            sn91bg_phase1 = False
            sn91bg_phase2 = False
            sn91bg_phase3 = False
            sn91bg_phase4 = True
            sn91bg_running = False ## it should finish up in time where this is okay
            sn91bg_progress = 'Finishing'
        elif sn91bg_phase3 or last_line.find('Finished writing') != -1:
            sn91bg_phase1 = False
            sn91bg_phase2 = False
            sn91bg_phase3 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            sn91bg_progress = '%.2f %%' % (float(completed) / float(total) * 60.0 + 40.0)
        elif sn91bg_phase2 or last_line.find('Read  (ised=') != -1:
            sn91bg_phase1 = False
            sn91bg_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n'][1]
            more_info = info.split('=')[-1].split(')')[0].split('/')
            total = more_info[1]
            completed = more_info[0]
            sn91bg_progress = '%.2f %%' % (float(completed) / float(total) * 40.0)
        elif sn91bg_phase1 == True:
            sn91bg_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i, Phase4: %i" %(int(sn91bg_phase1), int(sn91bg_phase2), int(sn91bg_phase3), int(sn91bg_phase4)))
            print("\tLast line: %s" %last_line)
            sn91bg_running = False
    else:
        sn91bg_progress = 'Done!'
        if sn91bg_error: sn91bg_progress = 'ERROR'

    """
    SNIa91bg-tr Progress Tracking
    """
    if sn91bg_tr_running:
        sn91bg_tr_log = open(sn91bg_tr_log_file, 'r')
        sn91bg_tr_info = sn91bg_tr_log.readlines()

        #check for errors and exit the loop if necessary
        for line in sn91bg_tr_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_tr_91bg.log")
                sn91bg_tr_running = False
                sn91bg_tr_error = True

        last_lines = sn91bg_tr_info[-10:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1 or line.find('Read  (ised=') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        
        sn91bg_tr_log.close()
        
        if sn91bg_tr_phase4 or (not sn91bg_tr_phase1 and not sn91bg_tr_phase2 and last_line.find('Finished writing') == -1):
            sn91bg_tr_phase1 = False
            sn91bg_tr_phase2 = False
            sn91bg_tr_phase3 = False
            sn91bg_tr_phase4 = True
            sn91bg_tr_running = False ## it should finish up in time where this is okay
            sn91bg_tr_progress = 'Finishing'
        elif sn91bg_tr_phase3 or last_line.find('Finished writing') != -1:
            sn91bg_tr_phase1 = False
            sn91bg_tr_phase2 = False
            sn91bg_tr_phase3 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            sn91bg_tr_progress = '%.2f %%' % (float(completed) / float(total) * 60.0 + 40.0)
        elif sn91bg_tr_phase2 or last_line.find('Read  (ised=') != -1:
            sn91bg_tr_phase1 = False
            sn91bg_tr_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n'][1]
            more_info = info.split('=')[-1].split(')')[0].split('/')
            total = more_info[1]
            completed = more_info[0]
            sn91bg_tr_progress = '%.2f %%' % (float(completed) / float(total) * 40.0)
        elif sn91bg_tr_phase1 == True:
            sn91bg_tr_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i, Phase4: %i" %(int(sn91bg_tr_phase1), int(sn91bg_tr_phase2), int(sn91bg_tr_phase3), int(sn91bg_tr_phase4)))
            print("\tLast line: %s" %last_line)
            sn91bg_tr_running = False
    else:
        sn91bg_tr_progress = 'Done!'
        if sn91bg_tr_error: sn91bg_tr_progress = 'ERROR'

    """
    SNIax Progress Tracking
    """
    if iax_running:
        iax_log = open(iax_log_file, 'r')
        iax_info = iax_log.readlines()

        #check for errors and exit the loop if necessary
        for line in iax_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_iax.log")
                iax_running = False
                iax_error = True

        last_lines = iax_info[-10:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1 or line.find('Read  (ised=') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        
        iax_log.close()
        
        if iax_phase4 or (not iax_phase1 and not iax_phase2 and last_line.find('Finished writing') == -1):
            iax_phase1 = False
            iax_phase2 = False
            iax_phase3 = False
            iax_phase4 = True
            iax_running = False ## it should finish up in time where this is okay
            iax_progress = 'Finishing'
        elif iax_phase3 or last_line.find('Finished writing') != -1:
            iax_phase1 = False
            iax_phase2 = False
            iax_phase3 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            iax_progress = '%.2f %%' % (float(completed) / float(total) * 20.0 + 80.0)
        elif iax_phase2 or last_line.find('Read  (ised=') != -1:
            iax_phase1 = False
            iax_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n'][1]
            more_info = info.split('=')[-1].split(')')[0].split('/')
            total = more_info[1]
            completed = more_info[0]
            iax_progress = '%.2f %%' % (float(completed) / float(total) * 80.0)
        elif iax_phase1 == True:
            iax_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i, Phase4: %i" %(int(iax_phase1), int(iax_phase2), int(iax_phase3), int(iax_phase4)))
            print("\tLast line: %s" %last_line)
            iax_running = False
    else:
        iax_progress = 'Done!'
        if iax_error: iax_progress = 'ERROR'

    """
    SNIax-tr Progress Tracking
    """
    if iax_tr_running:
        iax_tr_log = open(iax_tr_log_file, 'r')
        iax_tr_info = iax_tr_log.readlines()

        #check for errors and exit the loop if necessary
        for line in iax_tr_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_iax.log")
                iax_tr_running = False
                iax_tr_error = True

        last_lines = iax_tr_info[-10:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1 or line.find('Read  (ised=') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        
        iax_tr_log.close()
        
        if iax_tr_phase4 or (not iax_tr_phase1 and not iax_tr_phase2 and last_line.find('Finished writing') == -1):
            iax_tr_phase1 = False
            iax_tr_phase2 = False
            iax_tr_phase3 = False
            iax_tr_phase4 = True
            iax_tr_running = False ## it should finish up in time where this is okay
            iax_tr_progress = 'Finishing'
        elif iax_tr_phase3 or last_line.find('Finished writing') != -1:
            iax_tr_phase1 = False
            iax_tr_phase2 = False
            iax_tr_phase3 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            iax_tr_progress = '%.2f %%' % (float(completed) / float(total) * 20.0 + 80.0)
        elif iax_tr_phase2 or last_line.find('Read  (ised=') != -1:
            iax_tr_phase1 = False
            iax_tr_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n'][1]
            more_info = info.split('=')[-1].split(')')[0].split('/')
            total = more_info[1]
            completed = more_info[0]
            iax_tr_progress = '%.2f %%' % (float(completed) / float(total) * 80.0)
        elif iax_tr_phase1 == True:
            iax_tr_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i, Phase4: %i" %(int(iax_tr_phase1), int(iax_tr_phase2), int(iax_tr_phase3), int(iax_tr_phase4)))
            print("\tLast line: %s" %last_line)
            iax_tr_running = False
    else:
        iax_tr_progress = 'Done!'
        if iax_tr_error: iax_tr_progress = 'ERROR'

    """
    SNPIa Progress Tracking
    """
    if pia_running:
        pia_log = open(pia_log_file, 'r')
        pia_info = pia_log.readlines()

        #check for errors and exit the loop if necessary
        for line in pia_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_pia.log")
                pia_running = False
                pia_error = True

        last_lines = pia_info[-10:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1 or line.find('Read  (ised=') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        
        pia_log.close()
        
        if pia_phase4 or (not pia_phase1 and not pia_phase2 and last_line.find('Finished writing') == -1):
            pia_phase1 = False
            pia_phase2 = False
            pia_phase3 = False
            pia_phase4 = True
            pia_running = False ## it should finish up in time where this is okay
            pia_progress = 'Finishing'
        elif pia_phase3 or last_line.find('Finished writing') != -1:
            pia_phase1 = False
            pia_phase2 = False
            pia_phase3 = True
            info = [x for x in last_line.strip().split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            pia_progress = '%.2f %%' % (float(completed) / float(total) * 10.0 + 90.0)
        elif pia_phase2 or last_line.find('Read  (ised=') != -1:
            pia_phase1 = False
            pia_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n'][1]
            more_info = info.split('=')[-1].split(')')[0].split('/')
            total = more_info[1]
            completed = more_info[0]
            pia_progress = '%.2f %%' % (float(completed) / float(total) * 90.0)
        elif pia_phase1 == True:
            pia_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i, Phase4: %i" %(int(pia_phase1), int(pia_phase2), int(pia_phase3), int(pia_phase4)))
            print("\tLast line: %s" %last_line)
            pia_running = False
    else:
        pia_progress = 'Done!'
        if pia_error: pia_progress = 'ERROR'

    """
    SNPIa-tr Progress Tracking
    """
    if pia_tr_running:
        pia_tr_log = open(pia_tr_log_file, 'r')
        pia_tr_info = pia_tr_log.readlines()

        #check for errors and exit the loop if necessary
        for line in pia_tr_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_tr_pia.log")
                pia_tr_running = False
                pia_tr_error = True

        last_lines = pia_tr_info[-10:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1 or line.find('Read  (ised=') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        
        pia_tr_log.close()
        
        if pia_tr_phase4 or (not pia_tr_phase1 and not pia_tr_phase2 and last_line.find('Finished writing') == -1):
            pia_tr_phase1 = False
            pia_tr_phase2 = False
            pia_tr_phase3 = False
            pia_tr_phase4 = True
            pia_tr_running = False ## it should finish up in time where this is okay
            pia_tr_progress = 'Finishing'
        elif pia_tr_phase3 or last_line.find('Finished writing') != -1:
            pia_tr_phase1 = False
            pia_tr_phase2 = False
            pia_tr_phase3 = True
            info = [x for x in last_line.strip().split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            pia_tr_progress = '%.2f %%' % (float(completed) / float(total) * 10.0 + 90.0)
        elif pia_tr_phase2 or last_line.find('Read  (ised=') != -1:
            pia_tr_phase1 = False
            pia_tr_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n'][1]
            more_info = info.split('=')[-1].split(')')[0].split('/')
            total = more_info[1]
            completed = more_info[0]
            pia_tr_progress = '%.2f %%' % (float(completed) / float(total) * 90.0)
        elif pia_tr_phase1 == True:
            pia_tr_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i, Phase4: %i" %(int(pia_tr_phase1), int(pia_tr_phase2), int(pia_tr_phase3), int(pia_tr_phase4)))
            print("\tLast line: %s" %last_line)
            pia_tr_running = False
    else:
        pia_tr_progress = 'Done!'
        if pia_tr_error: pia_tr_progress = 'ERROR'

    """
    SLSN Progress Tracking
    """
    if slsn_running:
        slsn_log = open(slsn_log_file, 'r')
        slsn_info = slsn_log.readlines()

        #check for errors and exit the loop if necessary
        for line in slsn_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_slsn.log")
                slsn_running = False
                slsn_error = True

        last_lines = slsn_info[-10:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1 or line.find('Read  (ised=') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        
        slsn_log.close()
        
        if slsn_phase4 or (not slsn_phase1 and not slsn_phase2 and last_line.find('Finished writing') == -1):
            slsn_phase1 = False
            slsn_phase2 = False
            slsn_phase3 = False
            slsn_phase4 = True
            slsn_running = False ## it should finish up in time where this is okay
            slsn_progress = 'Finishing'
        elif slsn_phase3 or last_line.find('Finished writing') != -1:
            slsn_phase1 = False
            slsn_phase2 = False
            slsn_phase3 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            slsn_progress = '%.2f %%' % (float(completed) / float(total) * 80.0 + 20.0)
        elif slsn_phase2 or last_line.find('Read  (ised=') != -1:
            slsn_phase1 = False
            slsn_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n'][1]
            more_info = info.split('=')[-1].split(')')[0].split('/')
            try:
                total = more_info[1]
            except:
                print('\n' + last_line + '\n')
                total = 700
            completed = more_info[0]
            slsn_progress = '%.2f %%' % (float(completed) / float(total) * 20.0)
        elif slsn_phase1 == True:
            slsn_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i, Phase4: %i" %(int(slsn_phase1), int(slsn_phase2), int(slsn_phase3), int(slsn_phase4)))
            print("\tLast line: %s" %last_line)
            slsn_running = False
    else:
        slsn_progress = 'Done!'
        if slsn_error: slsn_progress = 'ERROR'

    """
    SLSN-tr Progress Tracking
    """
    if slsn_tr_running:
        slsn_tr_log = open(slsn_tr_log_file, 'r')
        slsn_tr_info = slsn_tr_log.readlines()

        #check for errors and exit the loop if necessary
        for line in slsn_tr_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_tr_slsn.log")
                slsn_tr_running = False
                slsn_tr_error = True

        last_lines = slsn_tr_info[-10:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1 or line.find('Read  (ised=') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        
        slsn_tr_log.close()
        
        if slsn_tr_phase4 or (not slsn_tr_phase1 and not slsn_tr_phase2 and last_line.find('Finished writing') == -1):
            slsn_tr_phase1 = False
            slsn_tr_phase2 = False
            slsn_tr_phase3 = False
            slsn_tr_phase4 = True
            slsn_tr_running = False ## it should finish up in time where this is okay
            slsn_tr_progress = 'Finishing'
        elif slsn_tr_phase3 or last_line.find('Finished writing') != -1:
            slsn_tr_phase1 = False
            slsn_tr_phase2 = False
            slsn_tr_phase3 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            slsn_tr_progress = '%.2f %%' % (float(completed) / float(total) * 80.0 + 20.0)
        elif slsn_tr_phase2 or last_line.find('Read  (ised=') != -1:
            slsn_tr_phase1 = False
            slsn_tr_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n'][1]
            more_info = info.split('=')[-1].split(')')[0].split('/')
            try:
                total = more_info[1]
            except:
                print('\n' + last_line + '\n')
                total = 700
            completed = more_info[0]
            slsn_tr_progress = '%.2f %%' % (float(completed) / float(total) * 20.0)
        elif slsn_tr_phase1 == True:
            slsn_tr_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i, Phase4: %i" %(int(slsn_tr_phase1), int(slsn_tr_phase2), int(slsn_tr_phase3), int(slsn_tr_phase4)))
            print("\tLast line: %s" %last_line)
            slsn_tr_running = False
    else:
        slsn_tr_progress = 'Done!'
        if slsn_tr_error: slsn_tr_progress = 'ERROR'

    """
    TDE Progress Tracking
    """
    if tde_running:
        tde_log = open(tde_log_file, 'r')
        tde_info = tde_log.readlines()

        #check for errors and exit the loop if necessary
        for line in tde_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_tde.log")
                tde_running = False
                tde_error = True

        last_lines = tde_info[-10:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1 or line.find('Read  (ised=') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        
        tde_log.close()
        
        if tde_phase4 or (not tde_phase1 and not tde_phase2 and last_line.find('Finished writing') == -1):
            tde_phase1 = False
            tde_phase2 = False
            tde_phase3 = False
            tde_phase4 = True
            tde_running = False ## it should finish up in time where this is okay
            tde_progress = 'Finishing'
        elif tde_phase3 or last_line.find('Finished writing') != -1:
            tde_phase1 = False
            tde_phase2 = False
            tde_phase3 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            tde_progress = '%.2f %%' % (float(completed) / float(total) * 80.0 + 20.0)
        elif tde_phase2 or last_line.find('Read  (ised=') != -1:
            tde_phase1 = False
            tde_phase2 = True
            try:
                info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n'][1]
                more_info = info.split('=')[-1].split(')')[0].split('/')
                total = more_info[1]
                completed = more_info[0]
                tde_progress = '%.2f %%' % (float(completed) / float(total) * 20.0)
            except:
                print('\n\n' + last_line + '\n\n')
                tde_progress = 'DEBUG'
        elif tde_phase1 == True:
            tde_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i, Phase4: %i" %(int(tde_phase1), int(tde_phase2), int(tde_phase3), int(tde_phase4)))
            print("\tLast line: %s" %last_line)
            tde_running = False
    else:
        tde_progress = 'Done!'
        if tde_error: tde_progress = 'ERROR'
        
    """
    TDE-tr Progress Tracking
    """
    if tde_tr_running:
        tde_tr_log = open(tde_tr_log_file, 'r')
        tde_tr_info = tde_tr_log.readlines()

        #check for errors and exit the loop if necessary
        for line in tde_tr_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_tr_tde.log")
                tde_tr_running = False
                tde_tr_error = True

        last_lines = tde_tr_info[-10:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1 or line.find('Read  (ised=') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        
        tde_tr_log.close()
        
        if tde_tr_phase4 or (not tde_tr_phase1 and not tde_tr_phase2 and last_line.find('Finished writing') == -1):
            tde_tr_phase1 = False
            tde_tr_phase2 = False
            tde_tr_phase3 = False
            tde_tr_phase4 = True
            tde_tr_running = False ## it should finish up in time where this is okay
            tde_tr_progress = 'Finishing'
        elif tde_tr_phase3 or last_line.find('Finished writing') != -1:
            tde_tr_phase1 = False
            tde_tr_phase2 = False
            tde_tr_phase3 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            tde_tr_progress = '%.2f %%' % (float(completed) / float(total) * 80.0 + 20.0)
        elif tde_tr_phase2 or last_line.find('Read  (ised=') != -1:
            tde_tr_phase1 = False
            tde_tr_phase2 = True
            try:
                info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n'][1]
                more_info = info.split('=')[-1].split(')')[0].split('/')
                total = more_info[1]
                completed = more_info[0]
                tde_tr_progress = '%.2f %%' % (float(completed) / float(total) * 20.0)
            except:
                print('\n\n' + last_line + '\n\n')
                tde_tr_progress = 'DEBUG'
        elif tde_tr_phase1 == True:
            tde_tr_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i, Phase4: %i" %(int(tde_tr_phase1), int(tde_tr_phase2), int(tde_tr_phase3), int(tde_tr_phase4)))
            print("\tLast line: %s" %last_line)
            tde_tr_running = False
    else:
        tde_tr_progress = 'Done!'
        if tde_tr_error: tde_tr_progress = 'ERROR'

    """
    BBH Progress Tracking
    """
    if bbh_running:
        bbh_log = open(bbh_log_file, 'r')
        bbh_info = bbh_log.readlines()

        #check for errors and exit the loop if necessary
        for line in bbh_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_bbh.log")
                bbh_running = False
                bbh_error = True

        last_lines = bbh_info[-30:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1 or line.find('Read  (ised=') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        
        #last_line = kn_info[-2]
        bbh_log.close()
        
        if bbh_phase4 or (not bbh_phase1 and not bbh_phase2 and last_line.find('Finished writing') == -1):
            bbh_phase1 = False
            bbh_phase2 = False
            bbh_phase3 = False
            bbh_phase4 = True
            bbh_running = False ## it should finish up in time where this is okay
            bbh_progress = 'Finishing'
        elif bbh_phase3 or last_line.find('Finished writing') != -1:
            bbh_phase1 = False
            bbh_phase2 = False
            bbh_phase3 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            bbh_progress = '%.2f %%' % (float(completed) / float(total) * 20.0 + 80.0)
        elif bbh_phase2 or last_line.find('Read  (ised=') != -1:
            bbh_phase1 = False
            bbh_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n'][1]
            more_info = info.split('=')[-1].split(')')[0].split('/')
            total = more_info[1]
            completed = more_info[0]
            if total == 'with':
                bbh_progress = 'Done!'
                bbh_running = False
            else:
                bbh_progress = '%.2f %%' % (float(completed) / float(total) * 80.0)
        elif bbh_phase1 == True:
            bbh_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i, Phase4: %i" %(int(bbh_phase1), int(bbh_phase2), int(bbh_phase3), int(bbh_phase4)))
            print("\tLast line: %s" %last_line)
            bbh_running = False
    else:
        bbh_progress = 'Done!'
        if bbh_error: kn_progress = 'ERROR'

    """
    BBH-tr Progress Tracking
    """
    if bbh_tr_running:
        bbh_tr_log = open(bbh_tr_log_file, 'r')
        bbh_tr_info = bbh_tr_log.readlines()

        #check for errors and exit the loop if necessary
        for line in bbh_tr_info[-30:]:
            if line.find('FATAL') != -1:
                print("WARNING: ERROR found in sim_tr_bbh.log")
                bbh_tr_running = False
                bbh_tr_error = True

        last_lines = bbh_tr_info[-30:]
        last_lines.reverse()
        found_last_line = False
        for line in last_lines:
            if line.find('Finished writing') != -1 or line.find('Read  (ised=') != -1:
                last_line = line
                found_last_line = True
                break
        if not found_last_line:
            last_line = last_lines[0]
        
        #last_line = kn_info[-2]
        bbh_tr_log.close()
        
        if bbh_tr_phase4 or (not bbh_tr_phase1 and not bbh_tr_phase2 and last_line.find('Finished writing') == -1):
            bbh_tr_phase1 = False
            bbh_tr_phase2 = False
            bbh_tr_phase3 = False
            bbh_tr_phase4 = True
            bbh_tr_running = False ## it should finish up in time where this is okay
            bbh_tr_progress = 'Finishing'
        elif bbh_tr_phase3 or last_line.find('Finished writing') != -1:
            bbh_tr_phase1 = False
            bbh_tr_phase2 = False
            bbh_tr_phase3 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n']
            total = info[4]
            completed = info[2]
            bbh_tr_progress = '%.2f %%' % (float(completed) / float(total) * 20.0 + 80.0)
        elif bbh_tr_phase2 or last_line.find('Read  (ised=') != -1:
            bbh_tr_phase1 = False
            bbh_tr_phase2 = True
            info = [x for x in last_line.split(' ') if x != '' and x != '\t' and x != '\n'][1]
            more_info = info.split('=')[-1].split(')')[0].split('/')
            total = more_info[1]
            completed = more_info[0]
            if total == 'with':
                bbh_tr_progress = 'Done!'
                bbh_tr_running = False
            else:
                bbh_tr_progress = '%.2f %%' % (float(completed) / float(total) * 80.0)
        elif bbh_tr_phase1 == True:
            bbh_tr_progress = 'Initializing'
        else:
            print("\nNot sure how we got here, last status was:")
            print("\tPhase1: %i, Phase2: %i, Phase3: %i, Phase4: %i" %(int(bbh_tr_phase1), int(bbh_tr_phase2), int(bbh_tr_phase3), int(bbh_tr_phase4)))
            print("\tLast line: %s" %last_line)
            bbh_tr_running = False
    else:
        bbh_tr_progress = 'Done!'
        if bbh_tr_error: bbh_tr_progress = 'ERROR'
    
    
    #output progress
    progress_map = {'AGN': agn_progress,
                    'KN': kn_progress,
                    'Ia': ia_progress,
                    'CC': cc_progress,
                    'CaRT': cart_progress,
                    'ILOT': ilot_progress,
                    'Mdwarf': mdwarf_progress,
                    'SN91bg': sn91bg_progress,
                    'Iax': iax_progress,
                    'PIa': pia_progress,
                    'SLSN': slsn_progress,
                    'TDE': tde_progress,
                    'BBH': bbh_progress,
                    'AGN-tr': agn_tr_progress,
                    'KN-tr': kn_tr_progress,
                    'Ia-tr': ia_tr_progress,
                    'CC-tr': cc_tr_progress,
                    'CaRT-tr': cart_tr_progress,
                    'ILOT-tr': ilot_tr_progress,
                    'Mdwarf-tr': mdwarf_tr_progress,
                    'SN91bg-tr': sn91bg_tr_progress,
                    'Iax-tr': iax_tr_progress,
                    'PIa-tr': pia_tr_progress,
                    'SLSN-tr': slsn_tr_progress,
                    'TDE-tr': tde_tr_progress,
                    'BBH-tr': bbh_tr_progress}

    name_map = {obj : progress_map[obj] for obj in sys.argv[2:]}
    
    progress = 'Simulating:  ' + ' | '.join([obj + ' -- ' + name_map[obj] for obj in sys.argv[2:]]) + ' |                             '
    #progress = 'Simulating:  AGN -- %s | KN -- %s | Ia -- %s | CC -- %s |                          ' %(agn_progress, kn_progress, ia_progress, cc_progress)
    sys.stdout.write('\r%s' %progress)
    sys.stdout.flush()

    running = agn_running or kn_running or ia_running or cc_running or cart_running or ilot_running or mdwarf_running or sn91bg_running or iax_running or pia_running or slsn_running or tde_running or bbh_running
    running = running or agn_tr_running or kn_tr_running or ia_tr_running or cc_tr_running or cart_tr_running or ilot_tr_running or mdwarf_tr_running or sn91bg_tr_running or iax_tr_running or iax_tr_running or pia_tr_running or slsn_tr_running or tde_tr_running or bbh_tr_running
    time.sleep(2)
    
#sys.stdout.write('\rSimulating:  AGN -- Done! | KN -- Done! |  Ia -- Done!  |  CC -- Done!  |                         \n')

display_map = {}
for obj in sys.argv[2:]:
    if name_map[obj] == 'ERROR':
        display_map[obj] = ' ERROR'
    else:
        display_map[obj] = '  -- Done!'

sys.stdout.write('\rSimulating:  ' + ' | '.join([obj + ' ' + display_map[obj] for obj in sys.argv[2:]]) + ' |                             \n')
sys.stdout.flush()


#output a list of all sims that finished successfully
good_sims = []
try:
    for k, v in name_map.iteritems():
        if v != 'ERROR':
            good_sims.append(k)
except:
    good_sims = ['AGN', 'SN91bg', 'CC', 'PIa', 'ILOT', 'CaRT', 'TDE', 'KN', 'SLSN', 'Mdwarf', 'Ia', 'Iax', 'BBH']
    good_sims += [x + '-tr' for x in good_sims]

stream = open('../events/%s/logs/monitor_all_sims_report.log' %event_name, 'w+')
stream.write(','.join(good_sims))
stream.close()
