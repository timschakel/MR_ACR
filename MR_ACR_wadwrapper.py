#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 09:25:48 2022

@author: tschakel
"""

__version__ = '20220525'
__author__ = 'tschakel'

import os

# for local prototyping, add path
#import sys
#sys.path.insert(0, 'C:/Users/Tim/github/wadqc')

# this will fail unless wad_qc is already installed
from wad_qc.module import pyWADinput

"""
if not 'MPLCONFIGDIR' in os.environ:
    import pkg_resources
    try:
        #only for matplotlib < 3 should we use the tmp work around, but it should be applied before importing matplotlib
        matplotlib_version = [int(v) for v in pkg_resources.get_distribution("matplotlib").version.split('.')]
        if matplotlib_version[0]<3:
            os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 
    except:
        os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 

import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
"""

import MR_ACR_lib

if __name__ == "__main__":
    data, results, config = pyWADinput()
    
    # Log which series are found
    data_series = data.getAllSeries()
    print("The following series are found:")
    for item in data_series:
        print(item[0]["SeriesDescription"].value+" with "+str(len(item))+" instances")
    
    """
    Perform the analysis of the ACR Phantom Test.
    There are 7 measurements described:
        1. Geometric Accuracy
        2. High Contrast Spatial Resolution
        3. Slice Thickness Accuracy
        4. Slice Position Accuracy
        5. Image Intensity Uniformity
        6. Percent-Signal Ghosting
        7. Low-Contrast Object Detectability
    """
    
    # read runtime parameters for module
    for name,action in config['actions'].items():
        if name == 'acqdatetime':
            MR_ACR_lib.acqdatetime(data, results, action)
        # elif name == 'geometry_z':
        #     MR_ACR_lib.geometry_z(data, results, action)
        # elif name == 'geometry_xy':
        #     MR_ACR_lib.geometry_xy(data, results, action)
        # elif name == 'resolution_t1':
        #     MR_ACR_lib.resolution_t1(data, results, action)
        # elif name == 'resolution_t2':
        #     MR_ACR_lib.resolution_t2(data, results, action)
        # elif name == 'slice_thickness':
        #     MR_ACR_lib.slice_thickness(data, results, action)
        elif name == 'slice_pos_t1':
            MR_ACR_lib.slice_position(data, results, action)
        # elif name == 'slice_pos_t2':
        #     MR_ACR_lib.slice_pos_t2(data, results, action)
        # elif name == 'image_intensity_uniformity':
        #     MR_ACR_lib.image_intensity_uniformity(data, results, action)
        # elif name == 'percent_signal_ghosting':
        #     MR_ACR_lib.percent_signal_ghosting(data, results, action)
        # elif name == 'low_contrast_object_detectability':
        #     MR_ACR_lib.low_contrast_object_detectability(data, results, action)

    results.write()