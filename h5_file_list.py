import os
import glob

target_list = ['TIC432549364', 'TIC127425841', 'TIC186599508', 'TIC9348006', 'TIC82452140']

dir = "/datax/users/obs/lmanunza/output_turbo_seti/20210603"

for target in target_list:
    print('Target: ', target)
    for file in os.listdir(dir):
        # Create a simple .lst file of the .h5 files in the data directory
        dat_list = sorted(glob.glob(os.path.join(dir, '*.h5')))
        # This writes the .h5 files into a .lst, as required by the find_event_pipeline:
        list_name = target + "_h5_files.lst"
        dat_list_path = os.path.join("/datax/users/obs/lmanunza/output_turbo_seti/20210603", list_name)
        with open(dat_list_path, 'w') as f:
            for dat_path in dat_list:
                if target in dat_path:
                    f.write(dat_path + '\n')
