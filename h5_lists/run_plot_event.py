import os
import new_plot_event

target_list = ['432549364', '127425841', '186599508', '9348006', '82452140']

filters = [2,3]

for filt in filters:

    for target in target_list:

        # csv = target+"_f2_snr10.csv"
        csv = f"{target}_f{str(filt)}_snr10_03.csv"
        lst = "TIC"+target+"_h5_files.lst"

        try:

            new_plot_event.plot_candidate_events_individually(csv, lst, target, 'OFF', 'blc01', filt, show=True, overwrite=False, offset=0)
        except:
            pass

    for target in target_list:
        try:
            # cmd = 'convert %s %s' % ('TIC'+target+'*.png','TIC'+target+'.pdf')
            cmd = 'convert %s %s' % (f"TIC{target}*_f{str(filt)}.png",f"TIC{target}_f{str(filt)}.pdf")
            os.system(cmd)
        except:
            pass
