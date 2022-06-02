#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 09:27:16 2022

@author: tschakel
"""

from wad_qc.modulelibs import wadwrapper_lib
import wad_qc.module.moduledata
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from MR_ACR_util import (find_z_length,find_xy_diameter,retrieve_ellipse_parameters,
                         check_resolution_peaks1,check_resolution_peaks2)

### Helper functions
def getValue(ds, label):
    """Return the value of a pydicom DataElement in Dataset identified by label.

    ds: pydicom Dataset
    label: dicom identifier, in either pydicom Tag object, string or tuple form.
    """
    if isinstance(label, str):
        try:
            # Assume form "0x0008,0x1030"
            tag = pydicom.tag.Tag(label.split(','))
        except ValueError:
            try:
                # Assume form "SeriesDescription"
                tag = ds.data_element(label).tag
            except (AttributeError, KeyError):
                # `label` string doesn't represent an element of the DataSet
                return None
    else:
        # Assume label is of form (0x0008,0x1030) or is a pydicom Tag object.
        tag = pydicom.tag.Tag(label)

    try:
        return str(ds[tag].value)
    except KeyError:
        # Tag doesn't exist in the DataSet
        return None


def isFiltered(ds, filters):
    """Return True if the Dataset `ds` complies to the `filters`,
    otherwise return False.
    """
    for tag, value in filters.items():
        if not str(getValue(ds, tag)) == str(value):
            # Convert both values to string before comparison. Reason is that
            # pydicom can return 'str', 'int' or 'dicom.valuerep' types of data.
            # Similarly, the user (or XML) supplied value can be of any type.
            return False
    return True


def applyFilters(series_filelist, filters):
    """Apply `filters` to the `series_filelist` and return the filtered list.

    First, convert `filters` from an ElementTree Element to a dictionary
    Next, create a new list in the same shape as `series_filelist`, but only
    include filenames for which isFiltered returns True.
    Only include sublists (i.e., series) which are non empty.
    """
    # Turn ElementTree element attributes and text into filters
    #filter_dict = {element.attrib["name"]: element.text for element in filters}
    filter_dict = filters

    filtered_series_filelist = []
    # For each series in the series_filelist (or, study):
    for instance_filelist in series_filelist:
        # Filter filenames within each series
        filtered_instance_filelist = [fn for fn in instance_filelist
                                      if isFiltered(
                pydicom.read_file(fn, stop_before_pixels=True), filter_dict)]
        # Only add the series which are not empty
        if filtered_instance_filelist:
            filtered_series_filelist.append(filtered_instance_filelist)

    return filtered_series_filelist

def acqdatetime(data, results, action):
    """
    Get the date and time of acquisition
    """
    params = action["params"]
    datetime_series = data.getSeriesByDescription(params["datetime_series_description"])
    dt = wadwrapper_lib.acqdatetime_series(datetime_series[0][0])
    results.addDateTime('AcquisitionDateTime', dt) 

def geometry_z(data,results,action):
    params = action["params"]
    filters = action["filters"]
    
    """
    1. Geometric Accuracy
    Determine the  length (z) of the phantom
    Use the localizer (survey) 
    """
    # Select the localizer series and determine z length
    loc_filter = {item:filters.get(item)for item in ["SeriesDescription"]}
    sag_filter = {item:filters.get(item)for item in ["0x2001,100b"]}
    data_loc = applyFilters(data.series_filelist, loc_filter)
    data_loc_sag = applyFilters(data_loc, sag_filter)
    dcmInfile_loc,pixeldataIn_loc,dicomMode = wadwrapper_lib.prepareInput(data_loc_sag[0],headers_only=False)
    image_data_z = np.transpose(pixeldataIn_loc[1,:,:]) # take central slice (check if data is always ordered the same?)
        
    z_length_mm,geometry_z_filename = find_z_length(image_data_z, float(dcmInfile_loc.info.PixelSpacing[0]), dcmInfile_loc.info.AcquisitionDate, params)
    
    # Collect results
    results.addFloat("Geometry Z length", z_length_mm)
    results.addObject("Geometry Z figure", geometry_z_filename)
    
def geometry_xy(data,results,action):
    params = action["params"]
    filters = action["filters"]
    """
    1. Geometric Accuracy
    Determine the  diameter (xy) of the phantom
    Use the T1 scan 
    """
    # Select the localizer series and determine z length
    series_filter = {item:filters.get(item)for item in ["SeriesDescription"]}
    data_series = applyFilters(data.series_filelist, series_filter)
    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(data_series[0],headers_only=False)
    image_data_xy = np.transpose(pixeldataIn[int(params['slicenumber'])-1,:,:]) # take slice-1 (0-index)

    x_diameter_mm, y_diameter_mm,x_center_px,y_center_px,geometry_xy_filename = find_xy_diameter(image_data_xy, float(dcmInfile.info.PixelSpacing[0]), dcmInfile.info.AcquisitionDate, params)
    
    # Collect results
    results.addFloat("Geometry X diameter", x_diameter_mm)
    results.addFloat("Geometry Y diameter", y_diameter_mm)
    results.addFloat("Geometry X center pix", x_center_px)
    results.addFloat("Geometry Y center pix", y_center_px)
    results.addObject("Geometry XY diameter", geometry_xy_filename)

def resolution_t1(data,results,action):
    params = action["params"]
    filters = action["filters"]
    """
    2. Geometric Accuracy
    Determine the high contrast spatial resolution
    Use the T1 scan 
    """
    series_filter = {item:filters.get(item)for item in ["SeriesDescription"]}
    data_series = applyFilters(data.series_filelist, series_filter)
    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(data_series[0],headers_only=False)
    image_data = np.transpose(pixeldataIn[int(params['slicenumber'])-1,:,:]) # take slice-1 (0-index)
    image_data_center = np.transpose(pixeldataIn[5,:,:]) # take slice-1 (0-index)
    
    # location of the resolution insert is defined wrt center of the phantom
    x_center_px, y_center_px = retrieve_ellipse_parameters(image_data_center, mask_air_bubble=True)[0:2]
        
    res_coordoffsets = np.array([
        [-24,30],
        [ -7,37],
        [  0,30],
        [ 16,37],
        [ 24,31],
        [ 39,37],]) #check and maybe correct?
    
    res_locs = np.zeros([6,2],dtype=int)
    res_locs[:,0] = res_coordoffsets[:,0] + int(x_center_px)
    res_locs[:,1] = res_coordoffsets[:,1] + int(y_center_px)
    
    bg_coordoffsets = [22,37] #location wrt center
    bg = image_data[bg_coordoffsets[0]+int(y_center_px):bg_coordoffsets[0]+int(y_center_px)+15,
                    bg_coordoffsets[1]+int(x_center_px):bg_coordoffsets[1]+int(x_center_px)+15] # 15x15 size
    mean_bg = np.mean(bg)
    
    resolution_resolved1 = check_resolution_peaks1(image_data, res_locs, mean_bg,params['bg_factor'])
    #resolution_resolved2 = check_resolution_peaks2(image_data, res_locs)
    
    # Show the resolution insert:
    res_full_coordoffsets = np.array([13,-63]) # wrt center of phantom
    res_full_size = np.array([50,125])
    image_res = image_data[res_full_coordoffsets[0]+int(y_center_px):res_full_coordoffsets[0]+int(y_center_px)+res_full_size[0],
                           res_full_coordoffsets[1]+int(x_center_px):res_full_coordoffsets[1]+int(x_center_px)+res_full_size[1] ]
    saveas = "Resolution_T1.png"
    plt.imshow(image_res,cmap='gray')
    plt.title("Resolution_T1")
    plt.axis("off")
    plt.savefig(saveas, dpi=300)
    
    # Collect results
    results.addBool("Resolution T1 HOR 1.1 passed", resolution_resolved1[0])
    results.addBool("Resolution T1 HOR 1.0 passed", resolution_resolved1[1])
    results.addBool("Resolution T1 HOR 0.9 passed", resolution_resolved1[2])
    results.addBool("Resolution T1 VER 1.1 passed", resolution_resolved1[3])
    results.addBool("Resolution T1 VER 1.0 passed", resolution_resolved1[4])
    results.addBool("Resolution T1 VER 0.9 passed", resolution_resolved1[5])
    results.addObject("Resolution T1", saveas)

def resolution_t2(data,results,action):
    params = action["params"]
    filters = action["filters"]
    """
    2. Geometric Accuracy
    Determine the high contrast spatial resolution
    Use the T1 scan 
    """
    series_filter = {item:filters.get(item)for item in ["SeriesDescription"]}
    type_filter = {item:filters.get(item)for item in ["ImageType"]}
    echo_filter = {item:filters.get(item)for item in ["EchoNumbers"]}
    data_series = applyFilters(data.series_filelist, series_filter)
    data_series_type = applyFilters(data_series, type_filter)
    data_series_type_echo = applyFilters(data_series_type, echo_filter)
    
    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(data_series_type_echo[0],headers_only=False)
    image_data = np.transpose(pixeldataIn[int(params['slicenumber'])-1,:,:]) # take slice-1 (0-index)
    image_data_center = np.transpose(pixeldataIn[5,:,:]) # take slice-1 (0-index)
    
    # location of the resolution insert is defined wrt center of the phantom
    x_center_px, y_center_px = retrieve_ellipse_parameters(image_data_center, mask_air_bubble=True)[0:2]
        
    res_coordoffsets = np.array([
        [-24,30],
        [ -7,37],
        [  0,30],
        [ 16,37],
        [ 24,31],
        [ 39,37],]) #check and maybe correct?
    
    res_locs = np.zeros([6,2],dtype=int)
    res_locs[:,0] = res_coordoffsets[:,0] + int(x_center_px)
    res_locs[:,1] = res_coordoffsets[:,1] + int(y_center_px)
    
    bg_coordoffsets = [22,37] #location wrt center
    bg = image_data[bg_coordoffsets[0]+int(y_center_px):bg_coordoffsets[0]+int(y_center_px)+15,
                    bg_coordoffsets[1]+int(x_center_px):bg_coordoffsets[1]+int(x_center_px)+15] # 15x15 size
    mean_bg = np.mean(bg)
    
    resolution_resolved1 = check_resolution_peaks1(image_data, res_locs, mean_bg,params['bg_factor'])
    #resolution_resolved2 = check_resolution_peaks2(image_data, res_locs)
    
    # Show the resolution insert:
    res_full_coordoffsets = np.array([13,-63]) # wrt center of phantom
    res_full_size = np.array([50,125])
    image_res = image_data[res_full_coordoffsets[0]+int(y_center_px):res_full_coordoffsets[0]+int(y_center_px)+res_full_size[0],
                           res_full_coordoffsets[1]+int(x_center_px):res_full_coordoffsets[1]+int(x_center_px)+res_full_size[1] ]
    saveas = "Resolution_T2.png"
    plt.imshow(image_res,cmap='gray')
    plt.title("Resolution_T2")
    plt.axis("off")
    plt.savefig(saveas, dpi=300)
    
    # Collect results
    results.addBool("Resolution T2 HOR 1.1 passed", resolution_resolved1[0])
    results.addBool("Resolution T2 HOR 1.0 passed", resolution_resolved1[1])
    results.addBool("Resolution T2 HOR 0.9 passed", resolution_resolved1[2])
    results.addBool("Resolution T2 VER 1.1 passed", resolution_resolved1[3])
    results.addBool("Resolution T2 VER 1.0 passed", resolution_resolved1[4])
    results.addBool("Resolution T2 VER 0.9 passed", resolution_resolved1[5])
    results.addObject("Resolution T2", saveas)
        
        
    