# turboSETI
## Quick guide 
This guide explains, step by step, the workflow of narrowband technosignature searches through the tool turboSETI. 
This workflow is the one used in Manunza et al. (https://doi.org/10.1016/j.actaastro.2025.04.007) for TESS targets and Galactic Center.

### Step 1: Run turboSETI (turbo_seti.py)
  The input files of turboSETI are filterbank files (.fil). Not necessary for the turboSETI run, but they should be organized in three ON-OFF pairs, which is the main approach to exclude false positives. This will be useful for the next steps. 
The output are .h5 files (just the .fil converted), .dat (containing all the hits found), and a log file.

#### Step 1.1: Make lists of .dat files (dat_file_list.py)
  A list of .dat files, organized by target, is needed for further steps.

#### Step 1.2: Make lists of .h5 files (h5_file_list.py)
  A list of .h5 files, organized by target, is needed for further steps.


### Step 2: Find Event (run_find_event.py)
  The script takes as input .dat lists and produces .csv with the desired parameters (filter level, SNR, etc..). 
  

### Step 3: Plot Event (run_plot_event.py)
  The script takes as input .csv files previously produced alongside the .h5 lists. The output are the .png waterfall plots, labelled by TIC, drift rate, frequency and filter level. Also a .pdf collecting all waterfall plots for a given target and filter is produced.  



