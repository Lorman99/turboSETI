import new_find_event_pipeline

dat_lists = ['TIC432549364_dat_files.lst', 'TIC127425841_dat_files.lst', 'TIC186599508_dat_files.lst', 'TIC9348006_dat_files.lst', 'TIC82452140_dat_files.lst']

for dat_list in dat_lists:
    try:
        new_find_event_pipeline.find_event_pipeline(2,10,dat_list,on_off_first='ON',number_in_sequence=6,saving=True,zero_drift_parameter=False,user_validation=False)
    except:
        pass
