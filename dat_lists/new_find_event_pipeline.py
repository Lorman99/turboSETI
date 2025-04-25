#!/usr/bin/env python

#All star names must be entered in order, all A stars
#then all B-stars. In the future, I may automate this.
#node string = ex. 'blc00'
#filter_level = 1, 2, or 3
#SNR = the SNR cut for the analysis ex. 10
#dat_file_list_string = path to list of .dat files for
#    all of the candidates, with filename
#fil_file_list_string = path to list of .fil files for
#    all of the candidates, with filename

#required packages and programs
import new_find_event
import pandas as pd

#required for updated_find_event
import time
import numpy as np
#import glob

def find_event_pipeline(filter_level,
                        SNR,
                        dat_file_list_string,
                        on_off_first='ON',
                        number_in_sequence=6,
                        saving=False,
                        zero_drift_parameter=True,
                        user_validation=False):
    print("************   BEGINNING FIND_EVENT PIPELINE   **************")
    print("Assuming start with the " + on_off_first + " observation.")

    #Opening list of files
    dat_file_list = open(dat_file_list_string).readlines()
    dat_file_list = [files.replace('\n','') for files in dat_file_list]
    dat_file_list = [files.replace(',','') for files in dat_file_list]
    n_files = len(dat_file_list)
    print("LIST: ", dat_file_list)
    print("There are " + str(len(dat_file_list)) + " total files in your filelist, " + dat_file_list_string)
    print("Therefore, looking for events in " + str(int(n_files/number_in_sequence)) + " on-off sets")
    print("with a minimum SNR of " + str(SNR))

    if filter_level == 1:
        print("present in the A source with no RFI rejection from the off-sources")
    if filter_level == 2:
        print("Present in at least one A source with RFI rejection from the off-sources")
    if filter_level == 3:
        print("Present in all A sources with RFI rejection from the off-sources")

    if zero_drift_parameter == False:
        print("not including signals with zero drift")
    if zero_drift_parameter == True:
        print("including signals with zero drift")
    if saving == False:
        print("not saving the output files")
    if saving == True:
        print("saving the output files")

    if user_validation == True:
        question = "Do you wish to proceed with these settings?"
        while "the answer is invalid":
            reply = str(input(question+' (y/n): ')).lower().strip()
            if reply[0] == 'y':
                break
            if reply[0] == 'n':
                return

    #Looping over n_files chunks.
    candidate_list = []
    for i in range(int(len(dat_file_list)/n_files)):
        file_sublist = dat_file_list[n_files*i:n_files*(i+1)]
        print ('file_sublist', file_sublist)
        if on_off_first == 'ON':
            name=file_sublist[0].split('TIC')[1]
            print ('filename', name)
            name=name.split('_')[0]
        if on_off_first == 'OFF':
            name=file_sublist[1].split('TIC')[1]
            print ('filename', name)
            name=name.split('_')[0]
        print('name', name)
        cand = new_find_event.find_events(file_sublist, SNR_cut=SNR, check_zero_drift=zero_drift_parameter, filter_threshold=filter_level, on_off_first=on_off_first, number_in_sequence=number_in_sequence)
        print ('cand', cand)
        if len(cand) > 0 or type(cand) != None:
            candidate_list.append(cand)
    if len(candidate_list) > 0:
        find_event_output_dataframe = pd.concat(candidate_list)
    else:
        "Sorry, no potential candidates with your given parameters :("
        find_event_output_dataframe = []

    print("ENDING PIPELINE")

    if saving == True:
        if zero_drift_parameter == True:
            filestring = name + '_f' + str(filter_level) + '_snr' + str(SNR) + '_zero' + '.csv'
        else:
            filestring = name + '_f' + str(filter_level) + '_snr' + str(SNR) + '.csv'

        find_event_output_dataframe.to_csv(filestring)


    return(find_event_output_dataframe)


### Example command
#find_event_pipeline(2,10,'TIC15863518_dat_files.lst',on_off_first='ON',number_in_sequence=6,saving=True,zero_drift_parameter=False,user_validation=False)
