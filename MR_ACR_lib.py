#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 09:27:16 2022

@author: tschakel
"""

from wad_qc.modulelibs import wadwrapper_lib
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
import MR_ACR_dcm_input
from MR_ACR_util import (find_z_length,find_xy_diameter,retrieve_ellipse_parameters,
                         check_resolution_peaks1,check_resolution_peaks2,find_fwhm,
                         detect_edges,mask_to_coordinates,find_min_and_max_intensity_region,
                         get_mean_circle_ROI,get_mean_rect_ROI, find_centre_lowcontrast,
                         find_circles,get_slice_position_error,find_center,find_radius)

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
    print(">>> 1. Geometric Accuracy: Z-length <<<")
    # Select the localizer series and determine z length
    loc_filter = {item:filters.get(item)for item in ["SeriesDescription"]}
    sag_filter = {item:filters.get(item)for item in ["0x2001,100b"]}
    data_loc = applyFilters(data.series_filelist, loc_filter)
    data_loc_sag = applyFilters(data_loc, sag_filter)
    dcmInfile_loc,pixeldataIn_loc,dicomMode = MR_ACR_dcm_input.prepareInput(data_loc_sag[0],headers_only=False)
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
    print(">>> 1. Geometric Accuracy: XY-diameter <<<")
    # Select the localizer series and determine z length
    series_filter = {item:filters.get(item)for item in ["SeriesDescription"]}
    data_series = applyFilters(data.series_filelist, series_filter)
    dcmInfile,pixeldataIn,dicomMode = MR_ACR_dcm_input.prepareInput(data_series[0],headers_only=False)
    image_data_xy = np.transpose(pixeldataIn[int(params['slicenumber'])-1,:,:]) # take slice-1 (0-index)

    xy_diameter_mm, x_center_px,y_center_px,geometry_xy_filename = find_xy_diameter(image_data_xy, float(dcmInfile.info.PixelSpacing[0]), params)
    
    # Collect results
    results.addFloat("Geometry XY diameter", xy_diameter_mm)
    results.addObject("Geometry XY figure", geometry_xy_filename)

def resolution(data,results,action):
    """
    2. High Contrast Spatial Resolution
    Determine the high contrast resolution
    """
    print(">>> 2. High Contrast Spatial Resolution T1 <<<")
    params = action["params"]
    filters = action["filters"]
    savename = "Resolution.png"
    
    """
    Use the T1 scan 
    """
    t1_series_filter = {"SeriesDescription":filters.get(item)for item in ["t1_series_description"]}
    t1_data_series = applyFilters(data.series_filelist, t1_series_filter)
    dcmInfile,pixeldataIn,dicomMode = MR_ACR_dcm_input.prepareInput(t1_data_series[0],headers_only=False)
    image_data_t1 = np.transpose(pixeldataIn[int(params['slicenumber'])-1,:,:]) # take slice-1 (0-index)
    image_data_t1_center = np.transpose(pixeldataIn[5,:,:]) # take slice-1 (0-index)
    
    #check and maybe correct? THIS WAS PIXELS ON THE 1.0 MM GRID: 3.0T DATA APPARANTLY USES 0.8 MM GRIDS, SO ALL VOXEL OFFSETS ARE INCORRECT
    res_coordoffsets = np.array([
        [-24,30],
        [ -7,37],
        [  0,30],
        [ 16,37],
        [ 24,31],
        [ 39,37],]) / dcmInfile.info.PixelSpacing
    bg_coordoffsets = np.array([22,37]) / dcmInfile.info.PixelSpacing #location wrt center
    
    # location of the resolution insert is defined wrt center of the phantom
    #x_center_px, y_center_px = retrieve_ellipse_parameters(image_data_t1_center, mask_air_bubble=True)[0:2]
    x_center_px, y_center_px = find_center(image_data_t1_center,params)
    res_locs = np.zeros([6,2],dtype=int)
    res_locs[:,0] = np.int0(res_coordoffsets[:,0] + x_center_px)
    res_locs[:,1] = np.int0(res_coordoffsets[:,1] + y_center_px)
    
    bg = image_data_t1[np.int0(bg_coordoffsets[0]+y_center_px):np.int0(bg_coordoffsets[0]+y_center_px)+15,
                       np.int0(bg_coordoffsets[1]+x_center_px):np.int0(bg_coordoffsets[1]+x_center_px)+15] # 15x15 size
    mean_bg = np.mean(bg)
    
    resolution_resolved1 = check_resolution_peaks1(image_data_t1, res_locs, mean_bg,params['bg_factor_t1'])
    #resolution_resolved2 = check_resolution_peaks2(image_data, res_locs)
    
    res_full_coordoffsets = np.array([13,-63]) / dcmInfile.info.PixelSpacing# wrt center of phantom
    res_full_size = np.array([50,125]) / dcmInfile.info.PixelSpacing
    image_res = image_data_t1[np.int0(res_full_coordoffsets[0]+y_center_px):np.int0(res_full_coordoffsets[0]+y_center_px)+np.int0(res_full_size[0]),
                              np.int0(res_full_coordoffsets[1]+x_center_px):np.int0(res_full_coordoffsets[1]+x_center_px)+np.int0(res_full_size[1]) ]

    fig,axs = plt.subplots(1,2)
    axs[0].imshow(image_res,cmap='gray')
    axs[0].set_title("Resolution T1")
    axs[0].axis("off")
    
    # Collect results
    results.addBool("Resolution T1 HOR 1.1 passed", resolution_resolved1[0])
    results.addBool("Resolution T1 HOR 1.0 passed", resolution_resolved1[1])
    results.addBool("Resolution T1 HOR 0.9 passed", resolution_resolved1[2])
    results.addBool("Resolution T1 VER 1.1 passed", resolution_resolved1[3])
    results.addBool("Resolution T1 VER 1.0 passed", resolution_resolved1[4])
    results.addBool("Resolution T1 VER 0.9 passed", resolution_resolved1[5])
    
    """
    Use the T2 scan 
    """
    print(">>> 2. High Contrast Spatial Resolution T2 <<<")
    t2_series_filter = {"SeriesDescription":filters.get(item)for item in ["t2_series_description"]}
    type_filter = {item:filters.get(item)for item in ["ImageType"]}
    echo_filter = {item:filters.get(item)for item in ["EchoNumbers"]}
    data_series = applyFilters(data.series_filelist, t2_series_filter)
    data_series_type = applyFilters(data_series, type_filter)
    data_series_type_echo = applyFilters(data_series_type, echo_filter)
    
    dcmInfile,pixeldataIn,dicomMode = MR_ACR_dcm_input.prepareInput(data_series_type_echo[0],headers_only=False)
    image_data_t2 = np.transpose(pixeldataIn[int(params['slicenumber'])-1,:,:]) # take slice-1 (0-index)
    image_data_t2_center = np.transpose(pixeldataIn[5,:,:]) # take slice-1 (0-index)
    
    #x_center_px, y_center_px = retrieve_ellipse_parameters(image_data_t2_center, mask_air_bubble=True)[0:2]
    x_center_px, y_center_px = find_center(image_data_t2_center,params)
    res_locs = np.zeros([6,2],dtype=int)
    res_locs[:,0] = np.int0(res_coordoffsets[:,0] + x_center_px)
    res_locs[:,1] = np.int0(res_coordoffsets[:,1] + y_center_px)
    
    bg = image_data_t2[np.int0(bg_coordoffsets[0]+y_center_px):np.int0(bg_coordoffsets[0]+y_center_px)+15,
                       np.int0(bg_coordoffsets[1]+x_center_px):np.int0(bg_coordoffsets[1]+x_center_px)+15] # 15x15 size
    mean_bg = np.mean(bg)
    
    resolution_resolved1 = check_resolution_peaks1(image_data_t2, res_locs, mean_bg,params['bg_factor_t2'])
    #resolution_resolved2 = check_resolution_peaks2(image_data, res_locs)
    
    image_res = image_data_t2[np.int0(res_full_coordoffsets[0]+y_center_px):np.int0(res_full_coordoffsets[0]+y_center_px)+np.int0(res_full_size[0]),
                              np.int0(res_full_coordoffsets[1]+x_center_px):np.int0(res_full_coordoffsets[1]+x_center_px)+np.int0(res_full_size[1]) ]

    axs[1].imshow(image_res,cmap='gray')
    axs[1].set_title("Resolution T2")
    axs[1].axis("off")
    fig.savefig(savename,dpi=300)
    
    # Collect results
    results.addBool("Resolution T2 HOR 1.1 passed", resolution_resolved1[0])
    results.addBool("Resolution T2 HOR 1.0 passed", resolution_resolved1[1])
    results.addBool("Resolution T2 HOR 0.9 passed", resolution_resolved1[2])
    results.addBool("Resolution T2 VER 1.1 passed", resolution_resolved1[3])
    results.addBool("Resolution T2 VER 1.0 passed", resolution_resolved1[4])
    results.addBool("Resolution T2 VER 0.9 passed", resolution_resolved1[5])
    results.addObject("Resolution", savename)
        

def slice_thickness(data, results, action):
    params = action["params"]
    filters = action["filters"]
    fig_filename = "slice_thickness_test.png"
    """
    3. Slice Thickness Accuracy
    Determine the slice thickness using the ramps in slice 1
    Use the T1 & T2 scan 
    """
    print(">>> 3. Slice Thickness Accuracy T1 <<<")
    t1_series_filter = {"SeriesDescription":filters.get(item)for item in ["t1_series_description"]}
    #load T1
    t1_data_series = applyFilters(data.series_filelist, t1_series_filter)
    dcmInfile,pixeldataIn,dicomMode = MR_ACR_dcm_input.prepareInput(t1_data_series[0],headers_only=False)
    x_res = float(dcmInfile.info.PixelSpacing[0])
    t1_image_data = np.transpose(pixeldataIn[int(params['slicenumber'])-1,:,:]) # take slice-1 (0-index)
    
    
    #use slice 6 because slice 1 has too much sturctues in it
    image_data_center = np.transpose(pixeldataIn[5,:,:]) # take slice-1 (0-index)
    # x_center_px, y_center_px = retrieve_ellipse_parameters(image_data_center, mask_air_bubble=True)[0:2]
    x_center_px, y_center_px = find_center(image_data_center,params)
    #x_center_px, y_center_px = find_center(image_data_center,params)
    x_center_px = int(x_center_px)
    y_center_px = int(y_center_px)
    
    #load T2
    print(">>> 3. Slice Thickness Accuracy T2 <<<")
    t2_series_filter = {"SeriesDescription":filters.get(item)for item in ["t2_series_description"]}
    type_filter = {item:filters.get(item)for item in ["ImageType"]}
    echo_filter = {item:filters.get(item)for item in ["EchoNumbers"]}
    data_series = applyFilters(data.series_filelist, t2_series_filter)
    data_series_type = applyFilters(data_series, type_filter)
    data_series_type_echo = applyFilters(data_series_type, echo_filter)
    
    
    dcmInfile,pixeldataIn,dicomMode = MR_ACR_dcm_input.prepareInput(data_series_type_echo[0],headers_only=False)
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
    axs[0,0].axis('off')
    
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
    axs[0,1].axis('off')
    
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
    print(">>> 5. Image Intensity Uniformity T1 <<<")
    #load T1
    t1_series_filter = {"SeriesDescription":filters.get(item)for item in ["t1_series_description"]}
    t1_data_series = applyFilters(data.series_filelist, t1_series_filter)
    dcmInfile,pixeldataIn,dicomMode = MR_ACR_dcm_input.prepareInput(t1_data_series[0],headers_only=False)
    x_res = float(dcmInfile.info.PixelSpacing[0])
    t1_image_data = np.transpose(pixeldataIn[int(params['slicenumber'])-1,:,:]) # take slice-1 (0-index)
    
    
    #use slice 6 because slice 1 has too much sturctues in it
    image_data_center = np.transpose(pixeldataIn[5,:,:]) # take slice-1 (0-index)
    #x_center_px, y_center_px = retrieve_ellipse_parameters(image_data_center, mask_air_bubble=True)[0:2]
    x_center_px, y_center_px = find_center(image_data_center,params)
    x_center_px = int(x_center_px)
    y_center_px = int(y_center_px)
    
    #load T2
    print(">>> 5. Image Intensity Uniformity T2 <<<")
    t2_series_filter = {"SeriesDescription":filters.get(item)for item in ["t2_series_description"]}
    type_filter = {item:filters.get(item)for item in ["ImageType"]}
    echo_filter = {item:filters.get(item)for item in ["EchoNumbers"]}
    data_series = applyFilters(data.series_filelist, t2_series_filter)
    data_series_type = applyFilters(data_series, type_filter)
    data_series_type_echo = applyFilters(data_series_type, echo_filter)
    
    dcmInfile,pixeldataIn,dicomMode = MR_ACR_dcm_input.prepareInput(data_series_type_echo[0],headers_only=False)
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
    axs[0,0].axis('off')
    
    axs[1,0].imshow(t1_image_data, cmap=plt.get_cmap("Greys_r"), vmin = t1_stats[2]+10.0, vmax = t1_stats[2]+20.0)
    axs[1,0].add_patch(Circle((x_center_px, y_center_px+3), radius = radius_large_ROI, fill=False, ec = 'r'))
    axs[1,0].add_patch(Circle(t1_stats[3], radius = radius_small_ROI, fill=False, ec = 'b'))
    axs[1,0].set_title('T1 min ROI')
    axs[1,0].axis('off')
    
    
    axs[0,1].imshow(t2_image_data, cmap=plt.get_cmap("Greys_r"), vmin = t2_stats[0]-1.0, vmax = t2_stats[0])
    axs[0,1].add_patch(Circle((x_center_px, y_center_px+3), radius = radius_large_ROI, fill=False, ec = 'r'))
    axs[0,1].add_patch(Circle(t2_stats[1], radius = radius_small_ROI, fill=False, ec = 'b'))
    axs[0,1].set_title('T2 max ROI')
    axs[0,1].axis('off')
    
    axs[1,1].imshow(t2_image_data, cmap=plt.get_cmap("Greys_r"), vmin = t2_stats[2]+10.0, vmax = t2_stats[2]+20.0)
    axs[1,1].add_patch(Circle((x_center_px, y_center_px+3), radius = radius_large_ROI, fill=False, ec = 'r'))
    axs[1,1].add_patch(Circle(t2_stats[3], radius = radius_small_ROI, fill=False, ec = 'b'))
    axs[1,1].set_title('T2 min ROI')    
    axs[1,1].axis('off')
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
    6. Percent Signal Ghosting
    Draw some ROI's inside and outside of the phantom
    """
    #load T1
    print(">>> 6. Percent Signal Ghosting <<<")
    t1_series_filter = {item:filters.get(item)for item in ["SeriesDescription"]}
    t1_data_series = applyFilters(data.series_filelist, t1_series_filter)
    dcmInfile,pixeldataIn,dicomMode = MR_ACR_dcm_input.prepareInput(t1_data_series[0],headers_only=False)
    x_res = float(dcmInfile.info.PixelSpacing[0])
    t1_image_data = np.transpose(pixeldataIn[int(params['slicenumber'])-1,:,:]) # take slice-1 (0-index)
    
    #use slice 6 because slice 1 has too much sturctues in it
    image_data_center = np.transpose(pixeldataIn[5,:,:]) # take slice-1 (0-index)
    #x_center_px, y_center_px, width, height = retrieve_ellipse_parameters(image_data_center, mask_air_bubble=True)[0:4]
    x_center_px, y_center_px = find_center(image_data_center,params)
    radius = find_radius(image_data_center,params)
    width = int(radius)
    height = int(radius)
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
    ax.axis('off')
    plt.savefig(fig_filename, dpi=300)
    
    # write results:
    results.addFloat("Ghosting Ratio", 100*ghosting_ratio) #report percentage
    results.addObject("Percent Signal Ghosting", fig_filename)
    
def slice_position(data,results,action):
    params = action["params"]
    filters = action["filters"]
    savename = "Slice Position Error.png"
    """
    4. Slice Position Accuracy
    Determine the slice position accuracy
    Use the T1 scan 
    """
    print(">>> 4. Slice Position Accuracy T1 <<<")
    t1_series_filter = {"SeriesDescription":filters.get(item)for item in ["t1_series_description"]}
    t1_data_series = applyFilters(data.series_filelist, t1_series_filter)
    dcmInfile,pixeldataIn,dicomMode = MR_ACR_dcm_input.prepareInput(t1_data_series[0],headers_only=False)
    image_data_t1_top = np.transpose(pixeldataIn[int(params['slicenumbertop'])-1,:,:]) # take slice-1 (0-index)
    image_data_t1_bot = np.transpose(pixeldataIn[int(params['slicenumberbot'])-1,:,:]) # take slice-1 (0-index)
    image_data_t1_center = np.transpose(pixeldataIn[5,:,:]) # take slice-1 (0-index)
      
    # offsets for the Slice Position insert
    # location of the Slice Position insert is defined wrt center of the phantom:
    #x_center_px, y_center_px = retrieve_ellipse_parameters(image_data_t1_center, mask_air_bubble=True)[0:2]
 
    x_center_px, y_center_px = find_center(image_data_t1_center,params)
    x_center_px = int(x_center_px) #vertical in  plots
    y_center_px = int(y_center_px) #horizontal in plots
    
    title_bottom_t1 = "T1 bottom"
    title_top_t1 = "T1 top"
    
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Slice Position errors')
    slice_pos_error_bot_t1 = get_slice_position_error(image_data_t1_bot,x_center_px,y_center_px,axs[0,0],title_bottom_t1,dcmInfile.info.PixelSpacing[0])
    slice_pos_error_top_t1 = get_slice_position_error(image_data_t1_top,x_center_px,y_center_px,axs[1,0],title_top_t1,dcmInfile.info.PixelSpacing[0])
    
    """
    Use the T2 scan 
    """
    print(">>> 4. Slice Position Accuracy T2 <<<")
    t2_series_filter = {"SeriesDescription":filters.get(item)for item in ["t2_series_description"]}
    type_filter = {item:filters.get(item)for item in ["ImageType"]}
    echo_filter = {item:filters.get(item)for item in ["EchoNumbers"]}
    data_series = applyFilters(data.series_filelist, t2_series_filter)
    data_series_type = applyFilters(data_series, type_filter)
    data_series_type_echo = applyFilters(data_series_type, echo_filter)
    
    dcmInfile,pixeldataIn,dicomMode = MR_ACR_dcm_input.prepareInput(data_series_type_echo[0],headers_only=False)
    image_data_t2_top = np.transpose(pixeldataIn[int(params['slicenumbertop'])-1,:,:]) # take slice-1 (0-index)
    image_data_t2_bot = np.transpose(pixeldataIn[int(params['slicenumberbot'])-1,:,:]) # take slice-1 (0-index)
    image_data_t2_center = np.transpose(pixeldataIn[5,:,:]) # take slice-1 (0-index)
    
    #x_center_px, y_center_px = retrieve_ellipse_parameters(image_data_t2_center, mask_air_bubble=True)[0:2]
    x_center_px, y_center_px = find_center(image_data_t2_center,params)
    x_center_px = int(x_center_px) #vertical in  plots
    y_center_px = int(y_center_px) #horizontal in plots
    
    title_bottom_t2 = "T2 bottom"
    title_top_t2 = "T2 top"
    
    slice_pos_error_bot_t2 = get_slice_position_error(image_data_t2_bot,x_center_px,y_center_px,axs[0,1],title_bottom_t2,dcmInfile.info.PixelSpacing[0])
    slice_pos_error_top_t2 = get_slice_position_error(image_data_t2_top,x_center_px,y_center_px,axs[1,1],title_top_t2,dcmInfile.info.PixelSpacing[0])
    
    # Collect results
    fig.savefig(savename, dpi=300)
    results.addFloat("Slice Position Error T1 slice1", slice_pos_error_bot_t1)
    results.addFloat("Slice Position Error T1 slice11", slice_pos_error_top_t1)
    results.addFloat("Slice Position Error T2 slice1", slice_pos_error_bot_t2)
    results.addFloat("Slice Position Error T2 slice11", slice_pos_error_top_t2)
    results.addObject("Slice Position Error", savename)
  
def low_contrast_object_detectability(data, results, action):
    params = action["params"]
    filters = action["filters"]
    fig_filenames = ["t1_s1_low_contrast_object_detectability.png",
                     "t1_s2_low_contrast_object_detectability.png",
                     "t1_s3_low_contrast_object_detectability.png",
                     "t1_s4_low_contrast_object_detectability.png",
                     "t2_s1_low_contrast_object_detectability.png",
                     "t2_s2_low_contrast_object_detectability.png",
                     "t2_s3_low_contrast_object_detectability.png",
                     "t2_s4_low_contrast_object_detectability.png"]
    """
    7. Low Contrast Object Detectability
    """
    #load T1
    print(">>> 7. Low Contrast Object Detectability T1 <<<")
    t1_series_filter = {"SeriesDescription":filters.get(item)for item in ["t1_series_description"]}
    t1_data_series = applyFilters(data.series_filelist, t1_series_filter)
    dcmInfile,pixeldataIn,dicomMode = MR_ACR_dcm_input.prepareInput(t1_data_series[0],headers_only=False)
    x_res = float(dcmInfile.info.PixelSpacing[0])
    t1_image_data = pixeldataIn[int(params['firstslice'])-1:int(params['lastslice']),:,:] # take slice-1 (0-index)
    
    
    #use slice 6 because slice 1 has too much sturctues in it
    image_data_center = np.transpose(pixeldataIn[5,:,:]) # take slice-1 (0-index)
    #x_center_px, y_center_px = retrieve_ellipse_parameters(image_data_center, mask_air_bubble=True)[0:2]
    x_center_px, y_center_px = find_center(image_data_center,params)
    x_center_px = int(x_center_px)
    y_center_px = int(y_center_px)
    
    #load T2
    print(">>> 7. Low Contrast Object Detectability T2 <<<")
    t2_series_filter = {"SeriesDescription":filters.get(item)for item in ["t2_series_description"]}
    type_filter = {item:filters.get(item)for item in ["ImageType"]}
    echo_filter = {item:filters.get(item)for item in ["EchoNumbers"]}
    data_series = applyFilters(data.series_filelist, t2_series_filter)
    data_series_type = applyFilters(data_series, type_filter)
    data_series_type_echo = applyFilters(data_series_type, echo_filter)
    
    dcmInfile,pixeldataIn,dicomMode = MR_ACR_dcm_input.prepareInput(data_series_type_echo[0],headers_only=False)
    t2_image_data = pixeldataIn[int(params['firstslice'])-1:int(params['lastslice']),:,:] # take slice-1 (0-index)
    
    t1_count_spokes = 0
    t1_fig, ((t1_ax1, t1_ax2), (t1_ax3, t1_ax4)) = plt.subplots(2,2)
    t1_axs = [t1_ax1, t1_ax2, t1_ax3, t1_ax4]
    angle_offset = 8*np.pi/180 # ~8 degrees increase from slice to slice
    total_offset = 3*angle_offset
    
    for i in np.arange(3,-1,-1):
        image_data = np.transpose(t1_image_data[i,:,:]) # take slice-1 (0-index)
        lco_cx, lco_cy, radius = find_centre_lowcontrast(image_data,float(params['canny_sigma']),float(params['canny_low_threshold']), x_res)
        angle_offset_slice = total_offset - i*angle_offset
        count_spokes, fig_to_save = find_circles(image_data[int(lco_cy-radius):int(lco_cy+radius),int(lco_cx-radius):int(lco_cx+radius)],radius, t1_axs[i], x_res, angle_offset_slice, params)
        t1_count_spokes += count_spokes 
        fig_to_save.savefig(fig_filenames[i], dpi= 300)
    t1_fig.savefig("T1_slices.png", dpi= 300)
    
    t2_count_spokes = 0
    t2_fig, ((t2_ax1, t2_ax2), (t2_ax3, t2_ax4)) = plt.subplots(2,2)
    t2_axs = [t2_ax1, t2_ax2, t2_ax3, t2_ax4]
    for i in np.arange(3,-1,-1):
        image_data = np.transpose(t2_image_data[i,:,:]) # take slice-1 (0-index)
        lco_cx, lco_cy, radius = find_centre_lowcontrast(image_data,float(params['canny_sigma']),float(params['canny_low_threshold']),x_res)
        angle_offset_slice = total_offset - i*angle_offset
        count_spokes, fig_to_save = find_circles(image_data[int(lco_cy-radius):int(lco_cy+radius),int(lco_cx-radius):int(lco_cx+radius)],radius, t2_axs[i], x_res, angle_offset_slice, params)
        #count_spokes, fig_to_save = find_circles(image_data[int(lco_cy-radius):int(lco_cy+radius),int(lco_cx-radius):int(lco_cx+radius)],radius,float(params['edge_sigma']), t2_axs[i], x_res, float(params['edge_low_threshold']), float(params['edge_high_threshold']), float(params['window_leveling']))
        t2_count_spokes += count_spokes 
        fig_to_save.savefig(fig_filenames[i+4], dpi= 300)
    t2_fig.savefig("T2_slices.png", dpi= 300)
    
    #breakpoint()
    results.addFloat("T1 number of counted spokes", t1_count_spokes)
    results.addFloat("T2 number of counted spokes", t2_count_spokes)
    results.addObject("T1 slices", "T1_slices.png")
    results.addObject("T2 slices", "T2_slices.png")
    results.addObject("T1 slice 1", fig_filenames[0])
    results.addObject("T1 slice 2", fig_filenames[1])
    results.addObject("T1 slice 3", fig_filenames[2])
    results.addObject("T1 slice 4", fig_filenames[3])
    results.addObject("T2 slice 1", fig_filenames[4])
    results.addObject("T2 slice 2", fig_filenames[5])
    results.addObject("T2 slice 3", fig_filenames[6])
    results.addObject("T2 slice 4", fig_filenames[7])