#!/usr/bin/env python
'''
Script to plot events
    ..author: Sofia Sheikh (ssheikhmsa@gmail.com)
    ..modified to work for GC data by Karen Perez (kip8@cornell.edu)
'''

#General packages import
import glob
import os
import sys
import pandas as pd
#import pylab as plt
import numpy as np
import logging; logging.disable(logging.CRITICAL);

#BL + my packages import
#import updated_find_event
import pandas as pd
#from blimpy import Filterbank
#from blimpy.filterbank import *
import blimpy as bl

#Plotting packages import
import matplotlib
import matplotlib.pyplot as plt

#preliminary plot arguments
fontsize=16
font = {'family' : 'DejaVu Sans',
'size' : fontsize}
MAX_IMSHOW_POINTS = (10096, 10096)

from astropy.time import Time
def make_waterfall_plots(filenames_list, target, off_target, drates, fvals, f_start,f_stop, node_string, filter_level, ion=False,epoch=None,bw=250.0, local_host='',plot_name='',save_pdf_plot=False,saving_fig=False,offset=0,dedoppler=False,**kwargs):
    ''' Makes waterfall plots per group of ON-OFF pairs (up to 6 plots.)
    '''

    #prepares for plotting
    #target=target[0].strip('"')
    #target=target.decode("utf-8")
    tg=target[0]

    #tg=tg.decode('utf-8')
    tg=tg.split('_')
    print("TG: ", tg)
    tg=tg[0]
    #tg=tg[:-1]
    target=tg
    #tg=tg.decode('utf-8')
    #print ('tg', tg)
    print ('off_target', off_target)
    print('Preparing to plot: ', target)
    matplotlib.rc('font', **font)
    if ion:
        plt.ion()

    #defines a minimum and maximum of... something
    min_val = 0
    max_val = 5.
    factor = 1e6
    units = 'Hz'

    #sets up the sub-plots
    n_plots = len(filenames_list)
    fig = plt.subplots(n_plots, sharex=True, sharey=True,figsize=(10, 2*n_plots))

    #finding plotting values range for the first panel (A1)
    fil = bl.Waterfall(filenames_list[0], f_start=f_start, f_stop=f_stop)

    filheader=fil.header
    f={}
    for key, value in filheader.items():
        f[key]=value
        #f[key.decode("utf-8")]=value
    filheader=f

    print ('filheader', filheader)
    t0 = filheader['tstart']
    print ('t0', t0)
    plot_f, plot_data = fil.grab_data(f_start=f_start, f_stop=f_stop)
    dec_fac_x, dec_fac_y = 1, 1
    print ('plot_data', plot_data)

    #rebinning data to plot correctly with fewer plots
    if plot_data.shape[0] > MAX_IMSHOW_POINTS[0]:
        dec_fac_x = plot_data.shape[0] / MAX_IMSHOW_POINTS[0]
        print ('dec_fac_x', dec_fac_x)
    if plot_data.shape[1] > MAX_IMSHOW_POINTS[1]:
        dec_fac_y =  plot_data.shape[1] /  MAX_IMSHOW_POINTS[1]
        print ('dec_fac_y', dec_fac_y)
    #insert rebin definition code from utils.py in blimpy and edit it because this is the problem
    d=plot_data
    n_x=dec_fac_x
    n_y=dec_fac_y

    """ Rebin data by averaging bins together

    Args:
    d (np.array): data
    n_x (int): number of bins in x dir to rebin into one
    n_y (int): number of bins in y dir to rebin into one

    Returns:
    d: rebinned data with shape (n_x, n_y)
    """

    if d.ndim == 2:
        if n_y is None:
            n_y = 1
        if n_x is None:
            n_x = 1
        print ('d.shape[0]', d.shape[0])
        print ('d.shape[1]', d.shape[1])
        print ('n_x', n_x)
        print ('n_y', n_y)
        d = d[:int(d.shape[0] // n_x) * n_x, :int(d.shape[1] // n_y) * n_y]
        d = d.reshape((d.shape[0] // n_x, n_x, d.shape[1] // n_y, n_y))
        d = d.mean(axis=3)
        d = d.mean(axis=1)
    elif d.ndim == 1:
        d = d[:int(d.shape[0] // n_x) * n_x]
        d = d.reshape((d.shape[0] // n_x, n_x))
        d = d.mean(axis=1)
    else:
        raise RuntimeError("Only NDIM <= 2 supported")
    plot_data=d
    print ('d', d)




    #plot_data = rebin(plot_data, dec_fac_x, dec_fac_y)

    #investigate intensity values for A1 (first panel)
    row_sums = plot_data.sum(axis=1)
    plot_data = plot_data / row_sums[:, np.newaxis]
    #plot_data = 10*np.log10(plot_data)
    A1_avg = np.median(plot_data)
    A1_max = plot_data.max()
    A1_std = np.std(plot_data)

    #defining more plot parameters
    delta_f = 0.000250
    epoch = filheader['tstart']
    mid_f = np.abs(f_start+f_stop)/2.
    drate_max = np.max(np.abs(drates))

    subplots = []

    #working out intensity scale
    if kwargs.get('clim', None) is None:
        vmin=0
        vmax=1.0
        #vmin=A1_avg-A1_std*min_val-2
        #vmax=A1_avg+max_val*A1_std
    else:
        vmin, vmax = kwargs['clim']
    print ('vmin', vmin)
    print ('vmax', vmax)

    #Filling in each subplot for the full plot
    for i,filename in enumerate(filenames_list):
        subplot = plt.subplot(n_plots,1,i+1)
        subplots.append(subplot)
        #if i==0:
        #    plt.text(0.9, 0.9, tg, bbox=dict(facecolor='red', alpha=0.5))
        fil = bl.Waterfall(filename, f_start=f_start, f_stop=f_stop)
        filheader=fil.header
        z={}
        for key, value in filheader.items():
            z[key]=value
            #z[key.decode("utf-8")]=value
        filheader=z

        try:
            this_plot = plot_waterfall(fil,f_start=f_start, f_stop=f_stop, drate=drate_max,
                                       vmin=vmin,vmax=vmax,**kwargs)
            for drate, fval in zip(drates, fvals):
                t_elapsed = Time(filheader['tstart'], format='mjd').unix - Time(t0, format='mjd').unix
                t_duration = (fil.n_ints_in_file -1)* filheader['tsamp']
                f_event = fval + drate / 1e6 * t_elapsed
                overlay_drift(f_event, drate, t_duration, offset)
        except:
            raise

    #Titling the plot
        if i == 0:
            srcname = "%s-%s , $\dot{\\nu}$=%2.3f Hzs$^{-1}$ , MJD:%5.5f" % (tg, off_target,  drate_max, t0)
            print ('srcname', srcname)
            plt.title(srcname)
    #Plot formatting
        if i < len(filenames_list)-1:
            plt.xticks(np.arange(f_start, f_stop, delta_f/4.), ['','','',''])

    #More plot formatting.
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    xloc = np.linspace(f_start, f_stop, 5)
    xticks = [round(loc_freq) for loc_freq in (xloc - mid_f)*factor]
    if np.max(xticks) > 1000:
        xticks = [xt/1000 for xt in xticks]
        units = 'kHz'
    plt.text(0.05, 5.8, target, transform=ax.transAxes, bbox=dict(facecolor='white'))
    plt.text(0.05, 4.8, off_target, transform=ax.transAxes, bbox=dict(facecolor='white'))
    plt.text(0.05, 3.8, target, transform=ax.transAxes, bbox=dict(facecolor='white'))
    plt.text(0.05, 2.8, off_target, transform=ax.transAxes, bbox=dict(facecolor='white'))
    plt.text(0.05, 1.8, target, transform=ax.transAxes, bbox=dict(facecolor='white'))
    plt.text(0.05, 0.8, off_target, transform=ax.transAxes, bbox=dict(facecolor='white'))
    plt.xticks(xloc, xticks)
    plt.xlabel("Relative Frequency [%s] from %f MHz"%(units,mid_f),fontdict=font)

    #Colorbar
    cax = fig[0].add_axes([0.94, 0.11, 0.03, 0.77])
    fig[0].colorbar(this_plot,cax=cax,label='Power [dB counts]')

    #Adjust plots
    plt.subplots_adjust(hspace=0,wspace=0)

    #save the figures
    #plt.text(0.9, 0.9, tg, bbox=dict(facecolor='red', alpha=0.5))
    plt.savefig(tg+'-'+off_target + '_dr_' + "{:0.2f}".format(drate_max) + '_freq_' "{:0.2f}".format(f_start) + "_f"+str(filter_level)+".png",
               bbox_inches='tight')

    return subplots

def overlay_drift(f_event, drate, t_duration, offset=0):
    if offset == 'auto':
        offset = - 0.2 * drate*t_duration
    plt.plot((f_event+offset/1e6, f_event+drate/1e6*t_duration+offset/1e6), (0, t_duration), c='#cc0000', ls='dashed', lw=2)

def plot_waterfall(fil, f_start=None, f_stop=None, drate=None, if_id=0, logged=True,cb=False,freq_label=False,MJD_time=False, **kwargs):
    """ Plot waterfall of data
    Args:
        f_start (float): start frequency, in MHz
        f_stop (float): stop frequency, in MHz
        logged (bool): Plot in linear (False) or dB units (True),
        cb (bool): for plotting the colorbar
        kwargs: keyword args to be passed to matplotlib imshow()
    """
    #prepare font
    matplotlib.rc('font', **font)

    #Get the data
    plot_f, plot_data = fil.grab_data(f_start=f_start, f_stop=f_stop)

    # Make sure waterfall plot is under 4k*4k
    dec_fac_x, dec_fac_y = 1, 1
    if plot_data.shape[0] > MAX_IMSHOW_POINTS[0]:
        dec_fac_x = plot_data.shape[0] / MAX_IMSHOW_POINTS[0]
    if plot_data.shape[1] > MAX_IMSHOW_POINTS[1]:
        dec_fac_y =  plot_data.shape[1] /  MAX_IMSHOW_POINTS[1]
    #insert rebin definition code from utils.py in blimpy and edit it because this is the problem
    d=plot_data
    n_x=dec_fac_x
    n_y=dec_fac_y

    """ Rebin data by averaging bins together

    Args:
    d (np.array): data
    n_x (int): number of bins in x dir to rebin into one
    n_y (int): number of bins in y dir to rebin into one

    Returns:
    d: rebinned data with shape (n_x, n_y)
    """

    if d.ndim == 2:
        if n_y is None:
            n_y = 1
        if n_x is None:
            n_x = 1
        print ('d.shape[0]', d.shape[0])
        print ('d.shape[1]', d.shape[1])
        print ('n_x', n_x)
        print ('n_y', n_y)
        d = d[:int(d.shape[0] // n_x) * n_x, :int(d.shape[1] // n_y) * n_y]
        d = d.reshape((d.shape[0] // n_x, n_x, d.shape[1] // n_y, n_y))
        d = d.mean(axis=3)
        d = d.mean(axis=1)
    elif d.ndim == 1:
        d = d[:int(d.shape[0] // n_x) * n_x]
        d = d.reshape((d.shape[0] // n_x, n_x))
        d = d.mean(axis=1)
    else:
        raise RuntimeError("Only NDIM <= 2 supported")
    plot_data=d
    print ('d', d)


    #plot_data = rebin(plot_data, dec_fac_x, dec_fac_y)

    if MJD_time:
        extent=(plot_f[0], plot_f[-1], fil.timestamps[-1], fil.timestamps[0])
    else:
        extent=(plot_f[0], plot_f[-1], (fil.timestamps[-1]-fil.timestamps[0])*24.*60.*60, 0.0)

    #plots and scales intensity
    kwargs['cmap'] = kwargs.get('cmap', 'viridis')
    kwargs['logged'] = True
    if kwargs['logged'] == True:
        plot_data_norm=[]
        for row in plot_data:
            max_in_row=np.max(row)
            norm_row=[]
            for i in row:
                norm_elem=float(i)/float(max_in_row)
                norm_row.append(norm_elem)
            plot_data_norm.append(norm_row)
        """max_list=[]
        for row in plot_data:
            max_in_row=np.max(row)
            max_list.append(max_in_row)
        max_value=np.max(max_list)

        plot_data_norm=[]
        for row in plot_data:
            norm_row=[]
            for i in row:
                norm_elem=float(i)/float(max_value)
                norm_row.append(norm_elem)
            plot_data_norm.append(norm_row)
        #print ('plot_data_norm', plot_data_norm)"""

        plot_data=plot_data_norm
        #row_sums = plot_data.sum(axis=1)
        #plot_data = plot_data / row_sums[:, np.newaxis]
        #print ('norm_plot_data', plot_data)
        #plot_data = 10*np.log10(plot_data)
        #print ('log_plot_data', plot_data)
        kwargs.pop('logged')

    #shows the waterfall plot
    this_plot = plt.imshow(plot_data,
        aspect='auto',
        rasterized=True,
        interpolation='nearest',
        extent=extent,
        **kwargs
    )
    #add colorbar
    if cb:
        plt.colorbar()

    #add plot labels
    #plt.text(0.5, 0.5, 'B01', transform=ax.transAxes)
    if freq_label:
        plt.xlabel("Frequency [Hz]",fontdict=font)
    if MJD_time:
        plt.ylabel("Time [MJD]",fontdict=font)
    else:
        plt.ylabel("Time [s]",fontdict=font)
    return this_plot

def plot_candidate_events_individually(full_candidate_event_dataframe, correct_fils, source_name, off_source_name, node_string, filter_level, show=False, overwrite=False, offset=0, **kwargs):
    trimmed_candidate_event_dataframe=pd.read_csv(full_candidate_event_dataframe)
    print ('full_candidate_event_dataframe', full_candidate_event_dataframe)
    #get only the events in the dataframe that are from the right target
    #full_candidate_event_dataframe=full_candidate_event_dataframe['Source']
    print ('trimmed_candidate_event_dataframe_Source', trimmed_candidate_event_dataframe.loc[:, ['Source']])
    #print ('full_candidate_event_dataframe_Source', full_candidate_event_dataframe['Source'])
    candidate_event_dataframe = trimmed_candidate_event_dataframe.loc[:, ['Source', 'Freq', 'DriftRate','FreqStart']]
    #candidate_event_dataframe = full_candidate_event_dataframe.loc[full_candidate_event_dataframe['Source'] == source_name]
    print ('full_candidate_event_dataframe', candidate_event_dataframe)

    #load in the data for each individual hit
    for i in range(0, len(candidate_event_dataframe)):
        candidate = candidate_event_dataframe.iloc[i]
        source_id = candidate['Source']
        f_mid = candidate['Freq']
        drate = candidate['DriftRate']

        off_source_id=off_source_name

        #load in the list of .fil files
        filelist = open(correct_fils).readlines()
        filelist = [files.replace('\n','') for files in filelist]
        #print ('dat_file_list', dat_file_list)
        filelist = [files.replace(',','') for files in filelist]
        #filelist = correct_fils
        print ('filelist', filelist)
        #calculate the length of the total ABABAB from the fil files' headers
        fil0 = bl.Waterfall(filelist[0], load_data=True)
        print ('fil0', fil0)
        fil0header=fil0.header
        print ('fil0header', fil0header)
        y={}
        for key, value in fil0header.items():
            y[key]=value
            #y[key.decode("utf-8")]=value
        fil0header=y
        print ('fil0header', fil0header)
        t0 = fil0header['tstart']

        fil6 = bl.Waterfall(filelist[-1], load_data=True)
        print ('fil6', fil6)
        fil6header=fil6.header
        z={}
        for key, value in fil6header.items():
            z[key]=value
            #z[key.decode("utf-8")]=value
        fil6header=z
        t6 = fil6header['tstart']
        t_elapsed = Time(t6, format='mjd').unix - Time(t0, format='mjd').unix + (fil6.n_ints_in_file -1) * fil6header['tsamp']


        #calculate the width of the plot based on making sure the full drift is visible
        bw = 2.4*abs(drate)/1e6 * t_elapsed
        bw = np.max((bw, 500./1e6))

        #Get start and stop frequencies based on midpoint and bandwidth
        f_start, f_stop = np.sort((f_mid-bw/2,  f_mid+bw/2))

        #Print useful values
        print('*************************************************')
        print('***     The Parameters for This Plot Are:     ***')
        print('*************************************************')
        print('Target = ', source_id)
        print('Bandwidth (MHz) = ', bw)
        print('Total Time Elapsed (s) = ', t_elapsed)
        print('Start Frequency (MHz) = ', f_start)
        print('Middle Frequency (MHz) = ', f_mid)
        print('Stop Frequency (MHz) = ', f_stop)
        print('Expected Drift (Hz/s) = ', drate)
        print('Filter level: ', filter_level)
        print('*************************************************')
        print('*************************************************')

        #Pass info to make_waterfall_plots() function
        subplots = make_waterfall_plots(filelist,
                                        [source_id],
                                        off_source_id,
                                        [drate],
                                        [f_mid],
                                        f_start,
                                        f_stop,
                                        node_string,
                                        filter_level,
                                        ion=False,
                                        epoch=None,
                                        local_host='',
                                        plot_name='',
                                        save_pdf_plot=True,
                                        saving_fig=True,
                                        offset=offset,
                                        **kwargs)
    return

###Example command
#plot_candidate_events_individually('15863518_f2_snr10.csv', 'TIC15863518_h5_files.lst', 'B05', 'B02', 'blc01', 2, show=True, overwrite=False, offset=0)
