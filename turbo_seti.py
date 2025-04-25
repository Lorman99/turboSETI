import os

direc_list=['/datax2/20210603/GUPPI']

for dir in direc_list:
    print('Directory: ', dir)
    for file in os.listdir(dir):
        file_extract_name=file.split('.')
        #if file.endswith('0000.h5') and 'BLOCsurve' in file and 'kband' in file:
        if file.endswith('0000.fil') and 'TIC' in file and 'raw' not in file:
            file_extract=file.split('.fil')[0]
            file_dat=file_extract + '.dat'
            print ('file_dat', file_dat)
            if os.path.isfile(file_dat) == False:
                print('file: ', file)
                try:
                    cmd='turboSETI -M 4 -s 10 -b y %s' % (dir+'/'+file)
                    os.system(cmd)
                except:
                    pass
