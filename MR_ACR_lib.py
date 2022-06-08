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
from matplotlib.patches import Rectangle,Circle
import matplotlib.lines as lines
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
from MR_ACR_util import (find_z_length,find_xy_diameter,retrieve_ellipse_parameters,
                         check_resolution_peaks1,check_resolution_peaks2,find_fwhm,
                         detect_edges,mask_to_coordinates,find_min_and_max_intensity_region,
                         get_mean_circle_ROI,get_mean_rect_ROI,find_centre_lowcontrast)

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
    Use the T2 scan 
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
        

def slice_thickness(data, results, action):
    params = action["params"]
    filters = action["filters"]
    fig_filename = "slice_thickness_test.png"
    """
    3. Slice Thickness Accuracy
    Determine the slice thickness using the ramps in slice 1
    Use the T1 & T2 scan 
    """
    t1_series_filter = {"SeriesDescription":filters.get(item)for item in ["t1_series_description"]}
    #load T1
    t1_data_series = applyFilters(data.series_filelist, t1_series_filter)
    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(t1_data_series[0],headers_only=False)
    x_res = float(dcmInfile.info.PixelSpacing[0])
    t1_image_data = np.transpose(pixeldataIn[int(params['slicenumber'])-1,:,:]) # take slice-1 (0-index)
    
    
    #use slice 6 because slice 1 has too much sturctues in it
    image_data_center = np.transpose(pixeldataIn[5,:,:]) # take slice-1 (0-index)
    x_center_px, y_center_px = retrieve_ellipse_parameters(image_data_center, mask_air_bubble=True)[0:2]
    x_center_px = int(x_center_px)
    y_center_px = int(y_center_px)
    
    #load T2
    t2_series_filter = {"SeriesDescription":filters.get(item)for item in ["t2_series_description"]}
    type_filter = {item:filters.get(item)for item in ["ImageType"]}
    echo_filter = {item:filters.get(item)for item in ["EchoNumbers"]}
    data_series = applyFilters(data.series_filelist, t2_series_filter)
    data_series_type = applyFilters(data_series, type_filter)
    data_series_type_echo = applyFilters(data_series_type, echo_filter)
    
    
    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(data_series_type_echo[0],headers_only=False)
    t2_image_data = np.transpose(pixeldataIn[int(params['slicenumber'])-1,:,:]) # take slice-1 (0-index)
    
    # T1 slice thickness determination (bounds are excluding last point)
    ramp_1 = t1_image_data[y_center_px-4:y_center_px-1,x_center_px-7:x_center_px+8] # only center part (figure 11)
    ramp_2 = t1_image_data[y_center_px+1:y_center_px+4,x_center_px-7:x_center_px+8]
    
    # not taking into account scanners that use max value for zero!
    mean_ramp_1 = np.mean(ramp_1)
    mean_ramp_2 = np.mean(ramp_2)
    mean_ramps = (mean_ramp_1 + mean_ramp_2)/2.0
    
    # find full width half maximum for the ramps 
    t1_fwhm_val = mean_ramps/2.0
    ramp_1 = t1_image_data[y_center_px-4:y_center_px-1,:] # now the entire length of scan
    ramp_2 = t1_image_data[y_center_px+1:y_center_px+4,:]


    fwhm_ramp_1, t1_upper_1, t1_lower_1 = find_fwhm(t1_fwhm_val, ramp_1, x_center_px)
    fwhm_ramp_2, t1_upper_2, t1_lower_2 = find_fwhm(t1_fwhm_val, ramp_2, x_center_px)
    
    t1_slice_thickness = 0.2*(fwhm_ramp_1*fwhm_ramp_2)/(fwhm_ramp_1+fwhm_ramp_2)*x_res
    
    # T2 slice thickness determination 
    ramp_1 = t2_image_data[y_center_px-4:y_center_px-1,x_center_px-7:x_center_px+8] # only center part (figure 11)
    ramp_2 = t2_image_data[y_center_px+1:y_center_px+4,x_center_px-7:x_center_px+8]
    
    mean_ramp_1 = np.mean(ramp_1)
    mean_ramp_2 = np.mean(ramp_2)
    mean_ramps = (mean_ramp_1 + mean_ramp_2)/2.0
    
    # find full width half maximum for the ramps 
    t2_fwhm_val = mean_ramps/2.0
    ramp_1 = t2_image_data[y_center_px-4:y_center_px-1,:] # now the entire length of scan
    ramp_2 = t2_image_data[y_center_px+1:y_center_px+4,:]


    fwhm_ramp_1, t2_upper_1, t2_lower_1 = find_fwhm(t2_fwhm_val, ramp_1, x_center_px)
    fwhm_ramp_2, t2_upper_2, t2_lower_2 = find_fwhm(t2_fwhm_val, ramp_2, x_center_px)
    
    t2_slice_thickness = 0.2*(fwhm_ramp_1*fwhm_ramp_2)/(fwhm_ramp_1+fwhm_ramp_2)*x_res
    
    # T1 plots
    fig,axs = plt.subplots(2,2)
    axs[0,0].imshow(t1_image_data, cmap=plt.get_cmap("Greys_r"))
    axs[0,0].add_patch(Rectangle((x_center_px-7, y_center_px+1), 15, 2,fc ='none', ec ='b', lw = 1) )
    axs[0,0].add_patch(Rectangle((x_center_px-7, y_center_px-4), 15, 2,fc ='none', ec ='r', lw = 1) )
    axs[0,0].set_title("ROI's in T1 image" )
    
    axs[1,0].imshow(t1_image_data[y_center_px-10:y_center_px+10,x_center_px-51:x_center_px+50], cmap=plt.get_cmap("Greys_r"), vmin = t1_fwhm_val-1.0, vmax = t1_fwhm_val)
    axs[1,0].axvline(x=50-t1_lower_1,color='red')
    axs[1,0].axvline(x=50-t1_lower_2,color='blue')
    axs[1,0].axvline(x=50+t1_upper_1,color='red')
    axs[1,0].axvline(x=50+t1_upper_2,color='blue')
    axs[1,0].set_title("FWHM's in T1 image")
    
    # T2 plots
    axs[0,1].imshow(t2_image_data, cmap=plt.get_cmap("Greys_r"), vmin = 0, vmax= np.max(t2_image_data))
    axs[0,1].add_patch(Rectangle((x_center_px-7, y_center_px+1), 15, 2,fc ='none', ec ='b', lw = 1) )
    axs[0,1].add_patch(Rectangle((x_center_px-7, y_center_px-4), 15, 2,fc ='none', ec ='r', lw = 1) )
    axs[0,1].set_title("ROI's in T2 image" )
    
    axs[1,1].imshow(t2_image_data[y_center_px-10:y_center_px+10,x_center_px-51:x_center_px+50], cmap=plt.get_cmap("Greys_r"), vmin = t2_fwhm_val-1.0, vmax = t2_fwhm_val)
    axs[1,1].axvline(x=50-t2_lower_1,color='red')
    axs[1,1].axvline(x=50-t2_lower_2,color='blue')
    axs[1,1].axvline(x=50+t2_upper_1,color='red')
    axs[1,1].axvline(x=50+t2_upper_2,color='blue')
    axs[1,1].set_title("FWHM's in T2 image")
    
    plt.savefig(fig_filename, dpi=300)
    
    # write results:
    results.addFloat("Slice Thickness T1", t1_slice_thickness)
    results.addFloat("Slice Thickness T2", t2_slice_thickness)
    results.addObject("Slice Thickness test", fig_filename)

def image_intensity_uniformity(data, results, action):
    params = action["params"]
    filters = action["filters"]
    fig_filename = "image_intensity_uniformity_test.png"
    """
    5. Image Intensity Uniformity
    Draw some ROI's and determine the highest and lowest signal intensity areas
    """
    #load T1
    t1_series_filter = {"SeriesDescription":filters.get(item)for item in ["t1_series_description"]}
    t1_data_series = applyFilters(data.series_filelist, t1_series_filter)
    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(t1_data_series[0],headers_only=False)
    x_res = float(dcmInfile.info.PixelSpacing[0])
    t1_image_data = np.transpose(pixeldataIn[int(params['slicenumber'])-1,:,:]) # take slice-1 (0-index)
    
    
    #use slice 6 because slice 1 has too much sturctues in it
    image_data_center = np.transpose(pixeldataIn[5,:,:]) # take slice-1 (0-index)
    x_center_px, y_center_px = retrieve_ellipse_parameters(image_data_center, mask_air_bubble=True)[0:2]
    x_center_px = int(x_center_px)
    y_center_px = int(y_center_px)
    
    #load T2
    t2_series_filter = {"SeriesDescription":filters.get(item)for item in ["t2_series_description"]}
    type_filter = {item:filters.get(item)for item in ["ImageType"]}
    echo_filter = {item:filters.get(item)for item in ["EchoNumbers"]}
    data_series = applyFilters(data.series_filelist, t2_series_filter)
    data_series_type = applyFilters(data_series, type_filter)
    data_series_type_echo = applyFilters(data_series_type, echo_filter)
    
    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(data_series_type_echo[0],headers_only=False)
    t2_image_data = np.transpose(pixeldataIn[int(params['slicenumber'])-1,:,:]) # take slice-1 (0-index)
    
    radius_large_ROI = np.sqrt(200/np.pi)*10/x_res # roi should be 195cm^2-205cm^2 -> r in cm -> r in mm -> r in voxels
    radius_small_ROI = np.sqrt(1/np.pi)*10/x_res
    large_circle = Circle((x_center_px, y_center_px+3), radius = radius_large_ROI, fill=False)
    
    #T1 stats
    t1_stats = find_min_and_max_intensity_region(t1_image_data, large_circle, radius_small_ROI)
    #percent integral uniformity (PIU)
    t1_PIU = 100 * (1 - (t1_stats[0]-t1_stats[2])/(t1_stats[0]+t1_stats[2]))
    
    #T2 stats 
    t2_stats = find_min_and_max_intensity_region(t2_image_data, large_circle, radius_small_ROI)
    t2_PIU = 100 * (1 - (t2_stats[0]-t2_stats[2])/(t2_stats[0]+t2_stats[2]))
    
    fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(t1_image_data, cmap=plt.get_cmap("Greys_r"), vmin = t1_stats[0]-1.0, vmax = t1_stats[0])
    axs[0,0].add_patch(Circle((x_center_px, y_center_px+3), radius = radius_large_ROI, fill=False, ec = 'r'))
    axs[0,0].add_patch(Circle(t1_stats[1], radius = radius_small_ROI, fill=False, ec = 'b'))
    axs[0,0].set_title('T1 max ROI')
    
    axs[1,0].imshow(t1_image_data, cmap=plt.get_cmap("Greys_r"), vmin = t1_stats[2]+10.0, vmax = t1_stats[2]+20.0)
    axs[1,0].add_patch(Circle((x_center_px, y_center_px+3), radius = radius_large_ROI, fill=False, ec = 'r'))
    axs[1,0].add_patch(Circle(t1_stats[3], radius = radius_small_ROI, fill=False, ec = 'b'))
    axs[1,0].set_title('T1 min ROI')
    
    
    axs[0,1].imshow(t2_image_data, cmap=plt.get_cmap("Greys_r"), vmin = t2_stats[0]-1.0, vmax = t2_stats[0])
    axs[0,1].add_patch(Circle((x_center_px, y_center_px+3), radius = radius_large_ROI, fill=False, ec = 'r'))
    axs[0,1].add_patch(Circle(t2_stats[1], radius = radius_small_ROI, fill=False, ec = 'b'))
    axs[0,1].set_title('T2 max ROI')
    
    axs[1,1].imshow(t2_image_data, cmap=plt.get_cmap("Greys_r"), vmin = t2_stats[2]+10.0, vmax = t2_stats[2]+20.0)
    axs[1,1].add_patch(Circle((x_center_px, y_center_px+3), radius = radius_large_ROI, fill=False, ec = 'r'))
    axs[1,1].add_patch(Circle(t2_stats[3], radius = radius_small_ROI, fill=False, ec = 'b'))
    axs[1,1].set_title('T2 min ROI')    
    plt.savefig(fig_filename, dpi=300)
    
    # write results:
    results.addFloat("PIU T1", t1_PIU)
    results.addFloat("PIU T2", t2_PIU)
    results.addObject("Image Intensity Uniformity test", fig_filename)
    
def percent_signal_ghosting(data, results, action):
    params = action["params"]
    filters = action["filters"]
    fig_filename = "percent_signal_ghosting_test.png"
    """
    5. Percent Signal Ghosting
    Draw some ROI's inside and outside of the phantom
    """
    #load T1
    t1_series_filter = {item:filters.get(item)for item in ["SeriesDescription"]}
    t1_data_series = applyFilters(data.series_filelist, t1_series_filter)
    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(t1_data_series[0],headers_only=False)
    x_res = float(dcmInfile.info.PixelSpacing[0])
    t1_image_data = np.transpose(pixeldataIn[int(params['slicenumber'])-1,:,:]) # take slice-1 (0-index)
    
    #use slice 6 because slice 1 has too much sturctues in it
    image_data_center = np.transpose(pixeldataIn[5,:,:]) # take slice-1 (0-index)
    x_center_px, y_center_px, width, height = retrieve_ellipse_parameters(image_data_center, mask_air_bubble=True)[0:4]
    width = int(width)
    height = int(height)
    x_center_px = int(x_center_px)
    y_center_px = int(y_center_px)
    
    # circle inside phantom
    radius_large_ROI = np.sqrt(200/np.pi)*10/x_res # roi should be 195cm^2-205cm^2 -> r in cm -> r in mm -> r in voxels
    large_circle = Circle((x_center_px, y_center_px+3), radius = radius_large_ROI, fill=False, ec='r')
    
    #rectangles outside phantom
    top_rect = Rectangle((x_center_px-33, y_center_px-height-20), 66, 15, fill=False, ec='b')
    bot_rect = Rectangle((x_center_px-33, y_center_px+height+5), 66, 15, fill=False, ec='g')
    left_rect = Rectangle((x_center_px-width-20, y_center_px-33), 15, 66, fill=False, ec='y')
    right_rect = Rectangle((x_center_px+width+5, y_center_px-33), 15, 66, fill=False, ec='m')
    
    
    phantom_mean_val = get_mean_circle_ROI(t1_image_data, large_circle)
    
    top_mean_val = get_mean_rect_ROI(t1_image_data, top_rect)
    bot_mean_val = get_mean_rect_ROI(t1_image_data, bot_rect)
    left_mean_val = get_mean_rect_ROI(t1_image_data, left_rect)
    right_mean_val = get_mean_rect_ROI(t1_image_data, right_rect)
    
    ghosting_ratio = np.abs(((top_mean_val + bot_mean_val) - (left_mean_val + right_mean_val))/(2*phantom_mean_val))
    
    fig, ax = plt.subplots()
    ax.imshow(t1_image_data, cmap=plt.get_cmap("Greys_r"))
    ax.add_patch(large_circle)
    ax.add_patch(top_rect)
    ax.add_patch(bot_rect)
    ax.add_patch(left_rect)
    ax.add_patch(right_rect)
    ax.set_title("T1 map with ROI's")
    plt.savefig(fig_filename, dpi=300)
    
    # write results:
    results.addFloat("Ghosting Ratio", ghosting_ratio)
    results.addObject("Percent Signal Ghosting", fig_filename)
    
def slice_pos_t1(data,results,action):
    params = action["params"]
    filters = action["filters"]
    """
    4. Slice Position Accuracy
    Determine the slice position accuracy
    Use the T1 scan 
    """
    series_filter = {item:filters.get(item)for item in ["SeriesDescription"]}
    data_series = applyFilters(data.series_filelist, series_filter)
    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(data_series[0],headers_only=False)
    image_data_top = np.transpose(pixeldataIn[int(params['slicenumbertop'])-1,:,:]) # take slice-1 (0-index)
    image_data_bot = np.transpose(pixeldataIn[int(params['slicenumberbot'])-1,:,:]) # take slice-1 (0-index)
    image_data_center = np.transpose(pixeldataIn[5,:,:]) # take slice-1 (0-index)
    
    # offsets for the Slice Position insert
    # location of the Slice Position insert is defined wrt center of the phantom:
    x_center_px, y_center_px = retrieve_ellipse_parameters(image_data_center, mask_air_bubble=True)[0:2]
    x_center_px = int(x_center_px)
    y_center_px = int(y_center_px)
    
    slice_offsets = [[-83,10],[-83,16]]
    x_range = 10
    y_range = 3
    
    edges_bot = detect_edges(image_data_bot)
    edges_bot_wedge1 = edges_bot[slice_offsets[0][0]+x_center_px:slice_offsets[0][0]+x_center_px+x_range,
                                 slice_offsets[0][1]+y_center_px:slice_offsets[0][1]+y_center_px+y_range]
    edges_bot_wedge2 = edges_bot[slice_offsets[1][0]+x_center_px:slice_offsets[1][0]+x_center_px+x_range,
                                 slice_offsets[1][1]+y_center_px:slice_offsets[1][1]+y_center_px+y_range]
    avg_ind_bot_edge1 = np.mean(np.argwhere(edges_bot_wedge1)[:,0])
    avg_ind_bot_edge2 = np.mean(np.argwhere(edges_bot_wedge2)[:,0])
    slice_pos_error_bot = avg_ind_bot_edge2 - avg_ind_bot_edge1
    
    edges_top = detect_edges(image_data_top)
    edges_top_wedge1 = edges_top[slice_offsets[0][0]+x_center_px:slice_offsets[0][0]+x_center_px+x_range,
                                 slice_offsets[0][1]+y_center_px:slice_offsets[0][1]+y_center_px+y_range]
    edges_top_wedge2 = edges_top[slice_offsets[1][0]+x_center_px:slice_offsets[1][0]+x_center_px+x_range,
                                 slice_offsets[1][1]+y_center_px:slice_offsets[1][1]+y_center_px+y_range]
    avg_ind_top_edge1 = np.mean(np.argwhere(edges_top_wedge1)[:,0])
    avg_ind_top_edge2 = np.mean(np.argwhere(edges_top_wedge2)[:,0])
    slice_pos_error_top = avg_ind_top_edge2 - avg_ind_top_edge1
    
    # Show the resolution insert:
    slice_pos_coordoffsets = np.array([-100,-21]) # wrt center of phantom
    slice_pos_size = np.array([50,40])
    image_slice_bot = image_data_bot[slice_pos_coordoffsets[0]+int(y_center_px):slice_pos_coordoffsets[0]+int(y_center_px)+slice_pos_size[0],
                                     slice_pos_coordoffsets[1]+int(x_center_px):slice_pos_coordoffsets[1]+int(x_center_px)+slice_pos_size[1] ]
    saveasbot = "Slice_position_bottom_T1.png"
    plt.figure(99)
    plt.imshow(image_slice_bot,cmap='gray')
    y1 = slice_offsets[0][0]+x_center_px+avg_ind_bot_edge1 - (slice_pos_coordoffsets[0]+y_center_px)
    y2 = slice_offsets[1][0]+x_center_px+avg_ind_bot_edge2 - (slice_pos_coordoffsets[0]+y_center_px)
    if int(np.sign(slice_pos_error_bot)) == 1:
        plt.axhline(y=y1,xmin=0.25,xmax=0.5,color='r')
        plt.axhline(y=y2,xmin=0.5,xmax=0.75,color='r')
    else:
        plt.axhline(y=y2,xmin=0.25,xmax=0.5,color='r')
        plt.axhline(y=y1,xmin=0.5,xmax=0.75,color='r')
    plt.title("Slice_position_bottom_T1")
    plt.axis("off")
    plt.savefig(saveasbot, dpi=300)
    
    image_slice_top = image_data_top[slice_pos_coordoffsets[0]+int(y_center_px):slice_pos_coordoffsets[0]+int(y_center_px)+slice_pos_size[0],
                                     slice_pos_coordoffsets[1]+int(x_center_px):slice_pos_coordoffsets[1]+int(x_center_px)+slice_pos_size[1] ]
    saveastop = "Slice_position_top_T1.png"
    plt.figure(98)
    plt.imshow(image_slice_top,cmap='gray')
    y1 = slice_offsets[0][0]+x_center_px+avg_ind_bot_edge1 - (slice_pos_coordoffsets[0]+y_center_px)
    y2 = slice_offsets[1][0]+x_center_px+avg_ind_bot_edge2 - (slice_pos_coordoffsets[0]+y_center_px)
    if int(np.sign(slice_pos_error_top)) == 1:
        plt.axhline(y=y1,xmin=0.25,xmax=0.5,color='r')
        plt.axhline(y=y2,xmin=0.5,xmax=0.75,color='r')
    else:
        plt.axhline(y=y2,xmin=0.25,xmax=0.5,color='r')
        plt.axhline(y=y1,xmin=0.5,xmax=0.75,color='r')
    plt.title("Slice_position_top_T1")
    plt.axis("off")
    plt.savefig(saveastop, dpi=300)
    
    # Collect results
    results.addFloat("Slice Position Error T1 slice1", slice_pos_error_bot)
    results.addFloat("Slice Position Error T1 slice11", slice_pos_error_top)
    results.addObject("Slice Position Error T1 slice1", saveasbot)
    results.addObject("Slice Position Error T1 slice11", saveastop)

def slice_pos_t2(data,results,action):
    params = action["params"]
    filters = action["filters"]
    """
    4. Slice Position Accuracy
    Determine the slice position accuracy
    Use the T2 scan 
    """
    series_filter = {item:filters.get(item)for item in ["SeriesDescription"]}
    type_filter = {item:filters.get(item)for item in ["ImageType"]}
    echo_filter = {item:filters.get(item)for item in ["EchoNumbers"]}
    data_series = applyFilters(data.series_filelist, series_filter)
    data_series_type = applyFilters(data_series, type_filter)
    data_series_type_echo = applyFilters(data_series_type, echo_filter)
    
    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(data_series_type_echo[0],headers_only=False)
    image_data_top = np.transpose(pixeldataIn[int(params['slicenumbertop'])-1,:,:]) # take slice-1 (0-index)
    image_data_bot = np.transpose(pixeldataIn[int(params['slicenumberbot'])-1,:,:]) # take slice-1 (0-index)
    image_data_center = np.transpose(pixeldataIn[5,:,:]) # take slice-1 (0-index)
    
    
    # offsets for the Slice Position insert
    # location of the Slice Position insert is defined wrt center of the phantom:
    x_center_px, y_center_px = retrieve_ellipse_parameters(image_data_center, mask_air_bubble=True)[0:2]
    x_center_px = int(x_center_px)
    y_center_px = int(y_center_px)
    
    slice_offsets = [[-83,10],[-83,16]]
    x_range = 10
    y_range = 3
    
    edges_bot = detect_edges(image_data_bot)
    edges_bot_wedge1 = edges_bot[slice_offsets[0][0]+x_center_px:slice_offsets[0][0]+x_center_px+x_range,
                                 slice_offsets[0][1]+y_center_px:slice_offsets[0][1]+y_center_px+y_range]
    edges_bot_wedge2 = edges_bot[slice_offsets[1][0]+x_center_px:slice_offsets[1][0]+x_center_px+x_range,
                                 slice_offsets[1][1]+y_center_px:slice_offsets[1][1]+y_center_px+y_range]
    avg_ind_bot_edge1 = np.mean(np.argwhere(edges_bot_wedge1)[:,0])
    avg_ind_bot_edge2 = np.mean(np.argwhere(edges_bot_wedge2)[:,0])
    slice_pos_error_bot = avg_ind_bot_edge2 - avg_ind_bot_edge1
    
    edges_top = detect_edges(image_data_top)
    edges_top_wedge1 = edges_top[slice_offsets[0][0]+x_center_px:slice_offsets[0][0]+x_center_px+x_range,
                                 slice_offsets[0][1]+y_center_px:slice_offsets[0][1]+y_center_px+y_range]
    edges_top_wedge2 = edges_top[slice_offsets[1][0]+x_center_px:slice_offsets[1][0]+x_center_px+x_range,
                                 slice_offsets[1][1]+y_center_px:slice_offsets[1][1]+y_center_px+y_range]
    avg_ind_top_edge1 = np.mean(np.argwhere(edges_top_wedge1)[:,0])
    avg_ind_top_edge2 = np.mean(np.argwhere(edges_top_wedge2)[:,0])
    slice_pos_error_top = avg_ind_top_edge2 - avg_ind_top_edge1
    
    # Show the resolution insert:
    slice_pos_coordoffsets = np.array([-100,-21]) # wrt center of phantom
    slice_pos_size = np.array([50,40])
    image_slice_bot = image_data_bot[slice_pos_coordoffsets[0]+int(y_center_px):slice_pos_coordoffsets[0]+int(y_center_px)+slice_pos_size[0],
                                     slice_pos_coordoffsets[1]+int(x_center_px):slice_pos_coordoffsets[1]+int(x_center_px)+slice_pos_size[1] ]
    saveasbot = "Slice_position_bottom_T2.png"
    plt.figure(97)
    plt.imshow(image_slice_bot,cmap='gray')
    y1 = slice_offsets[0][0]+x_center_px+avg_ind_bot_edge1 - (slice_pos_coordoffsets[0]+y_center_px)
    y2 = slice_offsets[1][0]+x_center_px+avg_ind_bot_edge2 - (slice_pos_coordoffsets[0]+y_center_px)
    if int(np.sign(slice_pos_error_bot)) == 1:
        plt.axhline(y=y1,xmin=0.25,xmax=0.5,color='r')
        plt.axhline(y=y2,xmin=0.5,xmax=0.75,color='r')
    else:
        plt.axhline(y=y2,xmin=0.25,xmax=0.5,color='r')
        plt.axhline(y=y1,xmin=0.5,xmax=0.75,color='r')
    plt.title("Slice_position_bottom_T2")
    plt.axis("off")
    plt.savefig(saveasbot, dpi=300)
    
    image_slice_top = image_data_top[slice_pos_coordoffsets[0]+int(y_center_px):slice_pos_coordoffsets[0]+int(y_center_px)+slice_pos_size[0],
                                     slice_pos_coordoffsets[1]+int(x_center_px):slice_pos_coordoffsets[1]+int(x_center_px)+slice_pos_size[1] ]
    saveastop = "Slice_position_top_T2.png"
    plt.figure(96)
    plt.imshow(image_slice_top,cmap='gray')
    y1 = slice_offsets[0][0]+x_center_px+avg_ind_bot_edge1 - (slice_pos_coordoffsets[0]+y_center_px)
    y2 = slice_offsets[1][0]+x_center_px+avg_ind_bot_edge2 - (slice_pos_coordoffsets[0]+y_center_px)
    if int(np.sign(slice_pos_error_top)) == 1:
        plt.axhline(y=y1,xmin=0.25,xmax=0.5,color='r')
        plt.axhline(y=y2,xmin=0.5,xmax=0.75,color='r')
    else:
        plt.axhline(y=y2,xmin=0.25,xmax=0.5,color='r')
        plt.axhline(y=y1,xmin=0.5,xmax=0.75,color='r')
    plt.title("Slice_position_top_T2")
    plt.axis("off")
    plt.savefig(saveastop, dpi=300)
    
    # Collect results
    results.addFloat("Slice Position Error T2 slice1", slice_pos_error_bot)
    results.addFloat("Slice Position Error T2 slice11", slice_pos_error_top)
    results.addObject("Slice Position Error T2 slice1", saveasbot)
    results.addObject("Slice Position Error T2 slice11", saveastop)
    
def lowcontrast_object_t1(data,results,action):
    params = action["params"]
    filters = action["filters"]
    """
    7. Low-contrast object detectability
    Use the T1 scan
    """
    series_filter = {item:filters.get(item)for item in ["SeriesDescription"]}
    data_series = applyFilters(data.series_filelist, series_filter)
    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(data_series[0],headers_only=False)
    
    # 1 slice for now, later loop over all 4
    angle_offset = 8*np.pi/180 #~ 8 degree rotation per slice
    for slice in np.arange(1,5):
        image_data = np.transpose(pixeldataIn[11-slice,:,:]) # take slice-1 (0-index)
        lco_cx, lco_cy, lco_radius = find_centre_lowcontrast(image_data,float(params['canny_sigma']),float(params['canny_low_threshold']))
        radius1 = 13
        radius2 = 26
        radius3 = 38   
        
        angles = np.linspace((-0.5*np.pi-(slice-1)*angle_offset),(1.5*np.pi-(slice-1)*angle_offset),num=100)
        #circ_coords = [lco_cx+lco_radius*np.cos(angles),lco_cy+lco_radius*np.sin(angles)]
        circ1_coords = [lco_cx+radius1*np.cos(angles),lco_cy+radius1*np.sin(angles)]
        circ2_coords = [lco_cx+radius2*np.cos(angles),lco_cy+radius2*np.sin(angles)]
        circ3_coords = [lco_cx+radius3*np.cos(angles),lco_cy+radius3*np.sin(angles)]
        
        x = np.arange(256)
        y = np.arange(256)
        f = interpolate.RectBivariateSpline(x, y, image_data)
        
        circ1_data = f.ev(np.array(circ1_coords[1]),np.array(circ1_coords[2]))
        circ2_data = f.ev(np.array(circ2_coords[0]),np.array(circ2_coords[1]))
        circ3_data = f.ev(np.array(circ3_coords[0]),np.array(circ3_coords[1]))
        
        circ1_data = image_data[np.array(circ1_coords[1]).astype(int),np.array(circ1_coords[0]).astype(int)] # interpolate ipv int?
        circ2_data = image_data[np.array(circ2_coords[1]).astype(int),np.array(circ2_coords[0]).astype(int)] # interpolate ipv int?
        circ3_data = image_data[np.array(circ3_coords[1]).astype(int),np.array(circ3_coords[0]).astype(int)] # interpolate ipv int?
        
        circ1_data_filt = gaussian_filter1d(circ1_data,1)
        circ2_data_filt = gaussian_filter1d(circ2_data,1)
        circ3_data_filt = gaussian_filter1d(circ3_data,1)

        breakpoint()
        
        peaks1,_ = find_peaks(circ1_data_filt,distance=8)
        peaks2,_ = find_peaks(circ2_data_filt,distance=8)
        peaks3,_ = find_peaks(circ3_data_filt,distance=8)
        
        fig, axs = plt.subplots(2,2)
        fig.suptitle('Results slice '+str(11-slice+1))
        axs[0,0].imshow(image_data,vmin=0.5*np.max(image_data),vmax=0.9*np.max(image_data),cmap='gray')
        axs[0,0].scatter(lco_cx,lco_cy)
        axs[0,0].scatter(circ1_coords[0],circ1_coords[1],s=1)
        axs[0,0].scatter(circ2_coords[0],circ2_coords[1],s=1)
        axs[0,0].scatter(circ3_coords[0],circ3_coords[1],s=1)
        #axs[0,0].axis([75,186,80,190])
        
        axs[0,1].plot(circ1_data)
        axs[0,1].plot(circ1_data_filt)
        axs[0,1].plot(peaks1,circ1_data_filt[peaks1],'x')
        axs[0,1].grid(True)
        axs[0,1].set_title('Signal & peaks inner ring')
        
        axs[1,0].plot(circ2_data)
        axs[1,0].plot(circ2_data_filt)
        axs[1,0].plot(peaks2,circ2_data_filt[peaks2],'x')
        axs[1,0].grid(True)
        axs[1,0].set_title('Signal & peaks middle ring')
        
        axs[1,1].plot(circ3_data)
        axs[1,1].plot(circ3_data_filt)
        axs[1,1].plot(peaks3,circ3_data_filt[peaks3],'x')
        axs[1,1].grid(True)
        axs[1,1].set_title('Signal & peaks outer ring')
   
    
 
    
    #TO DO: 
    # * offsets for the angles to generate the circ data 
    #   the 4 slices with the insert have an increasing angulation
    #   make sure the circular profile always starts right before the largest sphere and preceeds clockwise
    # * compare peak locations between inner/middle/outer circle
    #   when not at similar indices --> fail, spheres not detected for that spoke
    
   
    
   
    
   
    
   
    
   # plot stuff
   # fig, ax = plt.subplots()
   # make_circle = Circle((lco_cx, lco_cy), radius = lco_radius, fill=False, ec='r')
   # make_circle1 = Circle((lco_cx, lco_cy), radius = radius1, fill=False, ec='r')
   # make_circle2 = Circle((lco_cx, lco_cy), radius = radius2, fill=False, ec='g')
   # make_circle3 = Circle((lco_cx, lco_cy), radius = radius3, fill=False, ec='b')
   # make_line = lines.Line2D([lco_cx,lco_cx],[lco_cy,lco_cy-lco_radius])
   # ax.add_patch(make_circle)
   # ax.add_patch(make_circle1)
   # ax.add_patch(make_circle2)
   # ax.add_patch(make_circle3)
   # #ax.add_line(make_line)
   # ax.imshow(image_data,vmin=0.4*np.max(image_data),vmax=0.9*np.max(image_data))
   # ax.scatter(lco_cx,lco_cy)
   # plt.show()
    
   
    
   
    
   #  from scipy.fft import fft, fftfreq, fftshift
   #  N = 100
   #  T = 1
   #  yf = fft(circ1_data_filt)
   #  xf = fftfreq(N,T)
   #  xf = fftshift(xf)
   #  yplot = fftshift(yf)
   #  fig, ax = plt.subplots()
   #  plt.plot(xf,1.0/N * np.abs(yplot))

