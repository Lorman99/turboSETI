import os
import glob

target_list = ['TIC432549364', 'TIC127425841', 'TIC186599508', 'TIC9348006', 'TIC82452140']
dir = "/datax/users/obs/lmanunza/output_turbo_seti/20210603"

for target in target_list:
    print('Target: ', target)
    for file in os.listdir(dir):
        # Create a simple .lst file of the .dat files in the data directory
        dat_list = sorted(glob.glob(os.path.join(dir, '*.dat')))

        # This writes the .dat files into a .lst, as required by the find_event_pipeline:
        list_name = target + "_dat_files.lst"
        dat_list_path = os.path.join("/datax/users/obs/lmanunza/output_turbo_seti/20210603", list_name)
        with open(dat_list_path, 'w') as f:
            for dat_path in dat_list:
                if target in dat_path:
                    f.write(dat_path + '\n')

        # You don't have to print, but it's a good way to check that your list is in the correct order:
       # with open(dat_list_path, 'r') as f:
           # print(f.read())
