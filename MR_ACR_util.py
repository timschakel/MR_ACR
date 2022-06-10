#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 16:42:11 2022

@author: tschakel

To do:
    - plotting function: figure saving
    - weird notation from guido in function def..
    
"""
from pathlib import Path
from typing import List, Tuple, Any, Union
from skimage import feature
from skimage.transform import radon
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Circle
from scipy.signal import find_peaks
from scipy.interpolate import interp2d
import seaborn as sns

class bin_circle:
    def __init__(self, left, right, mean_val,origin,size):
        self.left = left 
        self.right = right
        self.mean_val = mean_val
        self.origin = origin #array
        self.angle = None #just initialize
        self.size = size
    
    def get_center(self):
        return ((self.right + self.left)/2)%self.size
    
    def get_rad(self):
        c = self.get_center()
        r1 = c - self.left 
        r2 = self.right - c
        return np.min([r1,r2])
    
    def set_angle(self,x1,x2):
        self.angle = np.arctan2(x1-self.origin[0], x2-self.origin[1])
        
    def get_angle(self):
        return self.angle

def detect_edges(
    image, sigma=0.3, low_threshold=750, high_threshold=None
) -> np.ndarray:
    """
    Detect edges on a 2d array
    :param high_threshold: high threshold for the hysteresis thresholding
    :param low_threshold: low threshold for the hysteresis thresholding
    :param sigma: width of the Gaussian
    :param image: 2d numpy array
    :return binary array of same dimensions as numpy_array representing the detected edges
    """

    # canny requires floats
    edges = feature.canny(
        image.astype("float32"),
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )

    return edges

def radon_transform(image: np.ndarray, max_deg: float = 180.0) -> np.ndarray:
    """Generate a sinogram for an image
    :param image: 2 dimensional data
    :param max_deg: maximum projection angle
    :return:
    """
    theta = np.linspace(0.0, max_deg, max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta)
    return sinogram

def interpolation_peak_offset(data: np.ndarray, peak_index: int) -> float:
    """Use the derivatives of the peak and its direct neighbours to find a "sub pixel" offset

    :param data: one dimensional, accessible by index, data structure
    :param peak_index: index for the peak
    :return: the approximate peak
    """
    # set x coordinates from -1 to +1 so the zero crossing can be added to the peak directly
    derived_1 = (-1, data[peak_index] - data[peak_index - 1])
    derived_2 = (1, data[peak_index + 1] - data[peak_index])

    slope = (derived_2[1] - derived_1[1]) / (derived_2[0] - derived_1[0])

    # y = mx + b --> b = y - mx
    offset = derived_1[1] - (slope * derived_1[0])
    # now solve 0 = slope*x + offset
    zero_point = -offset / slope

    return zero_point

def plot_edges_on_image(
    edges_x_y,
    acquisition,
    title=None,
    axlines=None,
    axvlines=None,
    axhlines=None,
    save_as=None,
):
    """

    :param axhlines: list of dicts containing axhlines params
    :param save_as:
    :param axvlines: list of dicts containing axvlines params
    :param edges_x_y: [[x coords],[y coords]]
    :param acquisition: pixel data as found on a pydicom object
    :param title: ...
    :param axlines:  list of dicts containing axlines params
    """
    fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})
    if edges_x_y:
        plt.scatter(edges_x_y[0], edges_x_y[1], s=1, c="r", alpha=0.3)
    if title:
        plt.title(title)
    plt.imshow(acquisition, cmap=plt.get_cmap("Greys_r"))

    if axlines:
        for line in axlines:
            plt.axline(**line)
    if axvlines:
        for line in axvlines:
            plt.axvline(**line)
    if axhlines:
        for line in axhlines:
            plt.axhline(**line)
    if save_as:
        plt.axis("off")
        plt.savefig(save_as, dpi=300)
    else:
        plt.show(block=True)

    plt.clf()
    plt.close()
    
def plot_ellipse_on_image(
    ellipse, acquisition, title=None, draw_axes=False, save_as: str = None):
    """
    :param ellipse: ellipse parameters [width, height, x center, y center, rotation]
    :param acquisition:
    :param title:
    :param draw_axes: whether or not to draw the axes
    :param save_as: absolute file path including extension. if used, a plot is not shown to the user
    """
    fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})
    ellipse_plot = Ellipse(
        xy=(ellipse[0], ellipse[1]),
        width=ellipse[2] * 2,
        height=ellipse[3] * 2,
        angle=ellipse[4],)
    
    ellipse_plot.set_edgecolor("red")
    ellipse_plot.set_linewidth(0.7)
    ellipse_plot.set_fill(False)
    ax.add_artist(ellipse_plot)
    if draw_axes:
        x_1, y_1 = [ellipse[0] - ellipse[2], ellipse[0] + ellipse[2]], [
            ellipse[1],
            ellipse[1],
        ]
        x_2, y_2 = [ellipse[0], ellipse[0]], [
            ellipse[1] - ellipse[3],
            ellipse[1] + ellipse[3],
        ]

        plt.plot(x_1, y_1, x_2, y_2, linewidth=0.5, color="r")
    if title:
        plt.title(title)
    plt.imshow(acquisition, cmap=plt.get_cmap("Greys_r"))

    if save_as:
        plt.axis("off")
        plt.savefig(save_as, dpi=300)
    else:
        plt.show(block=True)
    plt.clf()
    plt.close()
    
def mask_to_coordinates(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a boolean mask to x,y coordinates
    :param mask: boolean mask representing binary image of edges
    :return: tuple (np.array(x coordinates), np.array(y coordinates))
    """
    where = np.where(mask)

    y = where[0].astype(float)
    x = where[1].astype(float)

    return np.array(x), np.array(y)

def mask_edges(
    edges: np.ndarray,
    ellipse: List,
    removal_width: int = 60,  # TODO make removal_width configurable, make sure configurable items are in MM not in px!
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove edges in order to better fit the ellipse
    Pixels (that represent an edge) are removed based on two criteria
    :param removal_width: width in pixels of the upper part of the edges that will be set to zero due to the air bubble
    :param edges: x,y coordinates representing the edges of the phantom. ([x1,x2,...], [y1, y2,...])
    :param ellipse: parameters for an ellipse as returned by fit_ellipse
    :return: tuple containing an np.array for x and an np.array for y coordinates
    """
    # cast to regular list, as np.array is immutable
    edges_x = edges[0].tolist()
    edges_y = edges[1].tolist()

    center_x = ellipse[0]
    center_y = ellipse[1]

    edges_no_top_x, edges_no_top_y = _remove_top(
        edges_x, edges_y, removal_width, center_x, center_y
    )

    return np.array(edges_no_top_x), np.array(edges_no_top_y)

def _remove_top(
    x_coordinates: List, y_coordinates: List, removal_width: int, center_x, center_y
) -> Tuple[List, List]:
    """
    Remove top edges that are above center Y and between minus and plus half removal width of X
    :param center: x,y coordinates for the center of the ellipse
    :param edges: ([x1, x2,...], [y1, y2,...]) --> must be normal lists, because np.array is immutable
    :param removal_width: total width in pixels for removal. half of this will be on each side of center X
    :return:
    """

    half_removal_width = removal_width / 2
    removal_min = center_x - half_removal_width
    removal_max = center_x + half_removal_width

    indices_to_remove = []  # cannot remove while iterating, so keep track of indices
    for index, value in enumerate(x_coordinates):
        if (
            removal_min < value < removal_max and y_coordinates[index] < center_y
        ):  # y coordinates are reversed?
            indices_to_remove.append(index)

    indices_to_remove.sort(
        reverse=True
    )  # removing by index must be done from high to low to prevent index errors
    for index in indices_to_remove:
        del x_coordinates[index]
        del y_coordinates[index]

    return x_coordinates, y_coordinates

def retrieve_ellipse_parameters(image_data, mask_air_bubble=True):
    """

    :param mask_air_bubble: disable masking of the air bubble
    :param image_data: np.array
    :return: [major axis length, minor axis length, center x coordinate, center y coordinate, angle of rotation]
    """
    # obtain a binary image (mask) representing the edges on the image
    edges = detect_edges(image_data)
    # convert the mask to coordinates
    edge_coordinates = mask_to_coordinates(edges)
    # do a preliminary fitting of the ellipse
    # ellipse = fit_ellipse(edge_coordinates[0], edge_coordinates[1])
    from skimage.measure import EllipseModel
    # set coordinates to format that EllipseModel expects
    xy = np.array([[edge_coordinates[0][idx], edge_coordinates[1][idx]]
            for idx in range(len(edge_coordinates[0]))])
    
    ellipse_model = EllipseModel()
    if ellipse_model.estimate(xy):
        ellipse = ellipse_model.params
        # TODO else raise
    if mask_air_bubble:
        # create a new mask using the preliminary fit to remove air-bubble
        edge_coordinates_masked = mask_edges(edge_coordinates, ellipse)
        xy = np.array([[edge_coordinates_masked[0][idx], edge_coordinates_masked[1][idx]]
                for idx in range(len(edge_coordinates_masked[0]))])
        
        ellipse_model = EllipseModel()
        if ellipse_model.estimate(xy):
            ellipse = ellipse_model.params
            # TODO else raise
    return ellipse

def find_xy_diameter(image_data_xy,pixel_spacing, acqdate, params):
   """
   """
   [x_center_px,
    y_center_px,
    x_axis_length_px,
    y_axis_length_px,
    phi] = retrieve_ellipse_parameters(image_data_xy, mask_air_bubble=True)
   
   x_diameter_mm = x_axis_length_px * 2 * pixel_spacing
   y_diameter_mm = y_axis_length_px * 2 * pixel_spacing
   
   geometry_xy_filename = "geometry_xy_result.png"
   
   plot_ellipse_on_image(
       ellipse=[x_axis_length_px, y_axis_length_px, x_center_px, y_center_px, phi],
       acquisition=image_data_xy,
       title=f"XY Diameter, fitted ellipse, acq date: {acqdate}",
       draw_axes=True,
       save_as=geometry_xy_filename)
   
   return x_diameter_mm, y_diameter_mm,x_center_px,y_center_px,geometry_xy_filename  
    
def find_z_length(image_data_z,pixel_spacing,acqdate,params):
    """
    """
    edges_z = detect_edges(image_data_z, int(params['canny_sigma']), int(params['canny_low_threshold']))
    
    image_filtered = image_data_z.copy()
    # filter low signals out of the image
    image_filtered[image_filtered < image_filtered.max() / 4] = 0
    
    # the most top right pixel with a value is the approximate top right of the phantom
    phantom_top_right_column = np.where(np.sum(edges_z, axis=0) > 0)[0][-1]
    phantom_top_right_row = np.where(image_filtered[:,phantom_top_right_column] > 0)[0][0]
    phantom_top_right_xy = np.array([phantom_top_right_column, phantom_top_right_row])
    
    phantom_bottom_right_row = np.where(np.sum(edges_z, axis=1) > 0)[0][-1]
    phantom_bottom_right_column = np.where(image_filtered[phantom_bottom_right_row] > 0)[0][-1]
    phantom_bottom_right_xy = np.array([phantom_bottom_right_column, phantom_bottom_right_row])
    
    # the specified width of the phantom is 190mm, meaning the square root of 190**2 + side length**2 is the length of the diagonal
    phantom_right_side_length_px = np.linalg.norm(phantom_top_right_xy - phantom_bottom_right_xy)
    phantom_top_side_specified_width_px = 190 / pixel_spacing
    # Calculate (approximate) slope
    phantom_approximate_slope = (phantom_top_right_xy[0] - phantom_bottom_right_xy[0]) / (phantom_bottom_right_xy[1] - phantom_top_right_xy[1])
    
    # if the phantom was perpendicular resp. parallel to the x/y axes we can simply subtract the pixels for length and width
    # but because we have a (approximate) slope it is possible to correct
    x_delta = phantom_top_side_specified_width_px / 2
    y_delta = phantom_right_side_length_px / 2
    phantom_center_approximate_xy = np.array(
        [
            phantom_bottom_right_xy[0]
            - x_delta
            + (x_delta * phantom_approximate_slope),
            phantom_bottom_right_xy[1]
            - y_delta
            - (x_delta * phantom_approximate_slope),
        ]
    ).astype("int")
    
    pixel_range_for_cropping = round(45 / pixel_spacing)
    edges_z_crop = edges_z.copy()
    # crop vertically, set everything out side center +- 45 mm to 0
    edges_z_crop[:, : phantom_center_approximate_xy[0] - pixel_range_for_cropping] = 0
    edges_z_crop[:, phantom_center_approximate_xy[0] + pixel_range_for_cropping :] = 0

    # crop horizontally, set everything INSIDE center +- 45 mm to 0. This will make the resulting sinogram less noisy
    edges_z_crop[
        phantom_center_approximate_xy[1] - pixel_range_for_cropping : phantom_center_approximate_xy[1] + pixel_range_for_cropping,:] = 0
    
    sinogram = radon_transform(edges_z_crop)
    
    # find the coordinates for the two highest peaks
    # mask the sinogram so we only look at parallel, closest to perpendicular to columns
    sinogram_masked = sinogram.copy()
    sinogram_masked[:, :int(sinogram.shape[0]*0.4)] = 0
    sinogram_masked[:, int(sinogram.shape[0]*0.6):] = 0

    sino_max_per_row = np.sort(sinogram_masked)[
        :, sinogram.shape[0]-1
    ]  # this returns the final column (highest value per row) of the sorted rows of the sinogram
    
    # peak finder
    # find the column index for the first peak over 90% of the max value
    # this works well as we expect two high peaks in the data that are close to each other in value
    # a peak is where the following value is lower than the current
    threshold = 0.75
    peak_first = [
        (index, value)
        for index, value in enumerate(sino_max_per_row[:-1])
        if sino_max_per_row[index + 1] < value
        if value > np.max(sino_max_per_row) * threshold
    ][0]

    # find the column index for the last peak over 90% of the max value
    # same as previous, but in reverse order
    peak_last = [
        (len(sino_max_per_row) - index - 1, value)
        for index, value in enumerate(np.flip(sino_max_per_row)[:-1])
        if np.flip(sino_max_per_row)[index + 1] < value
        if value > np.max(sino_max_per_row) * threshold
    ][0]

    # take the peak values to match with the sinogram
    peak_1_value = peak_first[1]
    peak_2_value = peak_last[1]

    # this  yields two arrays where the first is the Y coordinate and the second is the X coordinate
    peak_1_coords = np.where(sinogram_masked == peak_1_value)
    peak_2_coords = np.where(sinogram_masked == peak_2_value)

    # rearrange into a format that is actually usable, interpolate the peaks
    peak_1_coords = np.array(
        [peak_1_coords[1][0],
         peak_1_coords[0][0] + interpolation_peak_offset(sino_max_per_row, peak_1_coords[0][0]),])
    peak_2_coords = np.array(
        [peak_2_coords[1][0],
         peak_2_coords[0][0] + interpolation_peak_offset(sino_max_per_row, peak_2_coords[0][0]),])

    # distance to center of sinogram is distance of line to center image
    sinogram_center = np.array([sinogram.shape[0] // 2, sinogram.shape[0] // 2])
    peak_1_center_distance_pixels = np.linalg.norm(sinogram_center - peak_1_coords)
    peak_2_center_distance_pixels = np.linalg.norm(sinogram_center - peak_2_coords)

    # length of phantom in mm is both lengths added and corrected for pixel spacing
    z_length_mm = (peak_1_center_distance_pixels + peak_2_center_distance_pixels) * pixel_spacing
    
    rotation_1 = peak_1_coords[0] * (180 / sinogram_masked.shape[0]) - 90
    rotation_2 = peak_2_coords[0] * (180 / sinogram_masked.shape[0]) - 90
    rotation = np.average([rotation_1, rotation_2])

    # for plotting we need the tangent of the (rotation converted to radians)
    slope = np.tan(np.radians(rotation))
    # save result objects to directory relative to working directory
    z_result_image_filename = "geometry_z_result.png"

    # use negative slope because of matplotlib stuffs
    # coordinates obtained for the peaks must be subtracted from the max index (i.e. shape) for plotting
    # TODO document all extra lines plotted
    plot_edges_on_image(
        mask_to_coordinates(edges_z),
        image_data_z,
        title=f"Z Length, Edges & Top/bottom/center highlighted, acq date: {acqdate}",
        axlines=[{"xy1": (sinogram_center[0],float(sinogram.shape[0] - peak_1_coords[1]),),
                  "slope": -slope,
                  "color": "b",
                  "linewidth": 0.75,},
                 {"xy1": (sinogram_center[0],float(sinogram.shape[0] - peak_2_coords[1]),),
                  "slope": -slope,
                  "color": "b",
                  "linewidth": 0.75,},],
        axvlines=[{"x": float(phantom_center_approximate_xy[0]),
                   "color": "y",
                   "linewidth": 0.75,},],
        axhlines=[{"y": float(phantom_center_approximate_xy[1]),
                   "color": "g",
                   "linewidth": 0.75,},],
        save_as=z_result_image_filename)
    
    return z_length_mm,z_result_image_filename
    
def check_resolution_peaks1(image_data, res_locs, mean_bg, bg_factor):
    """
    # Check the horizontal and vertical resolution
    # The grid has hole diameters of 1.1, 1.0, 0.9 mm
    # The spacing is twice the hole diameter
    # Use a search range of ~ 10 mm
    # Take horizontal/vertical profiles through the hole grids
    # If 4 peaks are found, the grid is considered resolved
    # (To avoid peaks from noise, the peaks need to be above 5 times the mean background signal)
    # If < 4 peaks are found, move to the next row/column in the grid
    """
    heigth = bg_factor * mean_bg
    resolution_resolved = [False, False, False, False, False, False]
    
    #mean_x_signal = np.mean(image_data[ res_locs[0,1]:res_locs[0,1]+y_range,
    #                                    res_locs[0,0]:res_locs[0,0]+x_range],axis=0)
    #mean_y_signal = np.mean(image_data[ res_locs[1,1]:res_locs[1,1]+y_range,
    #                                    res_locs[1,0]-x_range:res_locs[1,0]],axis=1)
    
    # Horizontal 1.1
    x_range = 12 #pixels
    y_range = 12 #pixels
    
    for y in range(y_range):
        x_signal = image_data[ res_locs[0,1]+y,
                               res_locs[0,0]:res_locs[0,0]+x_range]
        peaks,_ = find_peaks(x_signal,height=heigth)
        if len(peaks) == 4:
            print('Horizontal resolution 1.1 resolved')
            resolution_resolved[0] = True
            break
        
    # Horizontal 1.0
    x_range = 11 #pixels
    y_range = 11 #pixels
    for y in range(y_range):
        x_signal = image_data[ res_locs[2,1]+y,
                               res_locs[2,0]:res_locs[2,0]+x_range]
        peaks,_ = find_peaks(x_signal,height=heigth)
        if len(peaks) == 4:
            print('Horizontal resolution 1.0 resolved')
            resolution_resolved[1] = True
            break
        
    # Horizontal 0.9 --> smaller search range
    x_range = 10#pixels
    y_range = 10 #pixels
    for y in range(y_range):
        x_signal = image_data[ res_locs[4,1]+y,
                               res_locs[4,0]:res_locs[4,0]+x_range]
        peaks,_ = find_peaks(x_signal,height=heigth)
        if len(peaks) == 4:
            print('Horizontal resolution 0.9 resolved')
            resolution_resolved[2] = True
            break
    
    # Vertical 1.1
    x_range = 11 #pixels
    y_range = 11 #pixels
    for x in range(x_range):
        y_signal = image_data[ res_locs[1,1]:res_locs[1,1]+y_range,
                               res_locs[1,0]-x]
        peaks,_ = find_peaks(y_signal,height=heigth)
        if len(peaks) == 4:
            print('Vertical resolution 1.1 resolved')
            resolution_resolved[3] = True
            break
    
    # Vertical 1.0
    x_range = 11 #pixels
    y_range = 11 #pixels
    for x in range(x_range):
        y_signal = image_data[ res_locs[3,1]:res_locs[3,1]+y_range,
                               res_locs[3,0]-x]
        peaks,_ = find_peaks(y_signal,height=heigth)
        if len(peaks) == 4:
            print('Vertical resolution 1.0 resolved')
            resolution_resolved[4] = True
            break
        
    # Vertical 0.9
    x_range = 10 #pixels
    y_range = 10 #pixels
    
    for x in range(x_range):
        y_signal = image_data[ res_locs[5,1]:res_locs[5,1]+y_range,
                               res_locs[5,0]-x]
        peaks,_ = find_peaks(y_signal,height=heigth)
        if len(peaks) == 4:
            print('Vertical resolution 0.9 resolved')
            resolution_resolved[5] = True
            break
    
    # make a figure...
    return resolution_resolved
    
def check_resolution_peaks2(image_data, res_locs):
    """
    # Check the horizontal and vertical resolution
    # The grid has hole diameters of 1.1, 1.0, 0.9 mm
    # The spacing is twice the hole diameter
    # Use a search range of ~ 10 mm
    # Take the mean horizontal/vertical profile and find peaks
    # If 4 peaks are found, the grid is considered resolved
    """  
    resolution_resolved = np.zeros(6,dtype=int)
    
    #Horizontal 1.1
    x_range = 11 #pixels
    y_range = 11 #pixels
    x_signal = np.sum(image_data[ res_locs[0,1]:res_locs[0,1]+y_range,
                                  res_locs[0,0]:res_locs[0,0]+x_range],axis=0)
    peaks,_ = find_peaks(x_signal)
    if len(peaks) == 4:
        print('Horizontal resolution 1.1 resolved')
        resolution_resolved[0] = 1
       
    #Horizontal 1.0
    x_range = 11 #pixels
    y_range = 11 #pixels
    x_signal = np.sum(image_data[ res_locs[2,1]:res_locs[2,1]+y_range,
                                  res_locs[2,0]:res_locs[2,0]+x_range],axis=0)
    peaks,_ = find_peaks(x_signal)
    if len(peaks) == 4:
        print('Horizontal resolution 1.0 resolved')
        resolution_resolved[1] = 1
      
    #Horizontal 0.9
    x_range = 10 #pixels
    y_range = 10 #pixels
    x_signal = np.sum(image_data[ res_locs[4,1]:res_locs[4,1]+y_range,
                                  res_locs[4,0]:res_locs[4,0]+x_range],axis=0)
    peaks,_ = find_peaks(x_signal)
    if len(peaks) == 4:
        print('Horizontal resolution 0.9 resolved')
        resolution_resolved[2] = 1
    
    #Vertical 1.1
    x_range = 11 #pixels
    y_range = 11 #pixels
    y_signal = np.sum(image_data[ res_locs[1,1]:res_locs[1,1]+y_range,
                                  res_locs[1,0]-x_range:res_locs[1,0]],axis=1)
    peaks,_ = find_peaks(y_signal)
    if len(peaks) == 4:
        print('Vertical resolution 1.1 resolved')
        resolution_resolved[3] = 1
        
    #Vertical 1.0
    x_range = 11 #pixels
    y_range = 11 #pixels
    y_signal = np.sum(image_data[ res_locs[3,1]:res_locs[3,1]+y_range,
                                  res_locs[3,0]-x_range:res_locs[3,0]],axis=1)
    peaks,_ = find_peaks(y_signal)
    if len(peaks) == 4:
        print('Vertical resolution 1.0 resolved')
        resolution_resolved[4] = 1
        
    #Vertical 0.9
    x_range = 10 #pixels
    y_range = 10 #pixels
    y_signal = np.sum(image_data[ res_locs[5,1]:res_locs[5,1]+y_range,
                                  res_locs[5,0]-x_range:res_locs[5,0]],axis=1)
    peaks,_ = find_peaks(y_signal)
    if len(peaks) == 4:
        print('Vertical resolution 0.9 resolved')
        resolution_resolved[5] = 1
        
    # make a figure
    return resolution_resolved

def find_fwhm(val, ramp, xcenter):
    line = np.mean(ramp,axis=0)
    # x -> (idx, val) pair. find first idx where value is lower than val
    # xcenter -> end
    upper = next(x[0] for x in enumerate(line[xcenter:]) if x[1] < val)
    # xcenter -> begin
    lower = next(x[0] for x in enumerate(np.flip(line[0:xcenter])) if x[1] < val)
    
    fwhm = upper + lower
    return fwhm, upper, lower
        
def overlapping_circles(big, small):
    d = np.sqrt((small.center[0]-big.center[0])**2 + (small.center[1]-big.center[1])**2)
    return big.radius > (d + small.radius)

def point_in_circle(point, circle):
    return ((point[0]-circle.center[0])**2 + (point[1]-circle.center[1])**2) < circle.radius**2

def get_mean_circle_ROI(image, circle):
    tot_ROI = 0
    n_ROI = 0
    for y in range(circle.center[1]-int(np.ceil(circle.radius)), circle.center[1]+int(np.ceil(circle.radius))):
        for x in range(circle.center[0]-int(np.ceil(circle.radius)), circle.center[0]+int(np.ceil(circle.radius))):
            if point_in_circle((x, y), circle):
                tot_ROI += image[y,x]
                n_ROI += 1
    
    return tot_ROI/n_ROI

def find_min_and_max_intensity_region(image, LROI, small_radius):
    #LROI -> 200cm^2 circle, SROI -> 1cm^2
    start_point = (LROI.center[0]-int(np.ceil(LROI.radius)), LROI.center[1]-int(np.ceil(LROI.radius)))
    SROI = Circle(start_point, small_radius)
    # init empty values
    max_val = 0
    max_loc = (0, 0)
    min_val = np.max(image)
    min_loc = (0, 0)
    #loop in square region over the large circle
    #check if small circle is entirely inside
    #if true check value of region 
    #if larger than max or smaller than min set new values
    for y_offset in range(0,int(np.ceil(LROI.radius*2))):
        for x_offset in range(0,int(np.ceil(LROI.radius*2))):
            SROI.set_center((start_point[0] + x_offset, start_point[1] + y_offset))
            if overlapping_circles(LROI, SROI):
                mean_SROI = get_mean_circle_ROI(image, SROI)
                if mean_SROI > max_val:
                    max_val = mean_SROI
                    max_loc = SROI.center
                if mean_SROI < min_val:
                    min_val = mean_SROI
                    min_loc = SROI.center
    
    return max_val, max_loc, min_val, min_loc

"""
returns the mean value, but will also change the rect to be inside the domain of image
"""
def get_mean_rect_ROI(image, rect):
    bounds = image.shape
    if rect.get_x() < 0:
        rect.set_x(0)
    if rect.get_y() < 0:
        rect.set_y(0)
    if rect.get_x() + rect.get_width() > bounds[1]-1:
        rect.set_width(bounds[1]-1-rect.get_x())
    if rect.get_y() + rect.get_height() > bounds[0]-1:
        rect.set_height(bounds[0]-1-rect.get_y())
    
    mean_val = np.mean(image[rect.get_y():rect.get_y()+rect.get_height(),rect.get_x():rect.get_x()+rect.get_width()])
    return mean_val
        

def find_centre_lowcontrast(image_data,sigma,low_threshold):
    edges = feature.canny(image_data,
        sigma=sigma,low_threshold=low_threshold,high_threshold=None)

    searchradius = np.arange(42,47)
    hough_res = hough_circle(edges, searchradius)
    accums, cx, cy, radius = hough_circle_peaks(hough_res, searchradius, total_num_peaks=1)
    
    return cx, cy, radius

def create_circular_mask(image, rad):
    mask = np.zeros(image.shape,dtype=bool)
    circle = Circle((rad, rad), radius = rad-4)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if point_in_circle((y, x), circle):
                mask[y,x] = True
    
    return mask

def find_circles(image_data, rad, sigma, low_threshold):
    #interpolate image
    x = np.arange(0,image_data.shape[1])
    y = np.arange(0,image_data.shape[0])
    f = interp2d(y,x,image_data,kind='cubic')
    xnew = np.arange(0,image_data.shape[1], 0.25)
    ynew = np.arange(0,image_data.shape[0], 0.25)
    image_data_hr = f(ynew, xnew)
    rad*=4
    
    #get mask for edge detection
    mask = create_circular_mask(image_data_hr, rad)
    
    #do edge detection 
    edges = feature.canny(image_data_hr,
        sigma=4,low_threshold=8,high_threshold=15,mask = mask)
    edges = edges.astype('float64')

    #radii for circular profiles we take
    profile_radii = [52,102,152]
    
    fedges = interp2d(ynew*4,xnew*4,edges_float,kind='linear')
    fimage_hr = interp2d(ynew*4,xnew*4,image_data_hr,kind='cubic')

    angle_offset = 8*np.pi/180 #~ 8 degree rotation per slice    
    
    profile_coordinates = [] # coordinates of circular profile
    profile_edges = [] # edge detection data of the circular profile
    profile_data = [] # image data of the circular profile
    profile_edge_idxs = [] # the indices in profile_edges that are larger than 0 > edges
    profile_bins = [] # the circular profile divided into bins based on the edge data
    for circ in range(3):    
        angles = np.linspace((-0.5*np.pi-angle_offset),(1.5*np.pi-angle_offset),num=int(300*profile_radii[circ]/profile_radii[0]))
        profile_coordinates.append([rad+profile_radii[circ]*np.cos(angles),rad+profile_radii[circ]*np.sin(angles)])
        profile_edges.append([fedges(profile_coordinates[circ][0][i], profile_coordinates[circ][1][i])[0] for i in range(profile_coordinates[circ][0].shape[0])])
        profile_data.append([fimage_hr(profile_coordinates[circ][0][i], profile_coordinates[circ][1][i])[0] for i in range(profile_coordinates[circ][0].shape[0])])
        #normalize
        profile_data[circ] /= np.max(profile_data[circ])
        profile_edge_idxs.append([i for i,x in enumerate(profile_edges[circ]) if x > 0.0])
        #divide circle into bins    
        tmp_bins = []    
        for idx in range(1,len(profile_edge_idxs[circ])):
            if (profile_edge_idxs[circ][idx]-profile_edge_idxs[circ][idx-1]) > 1:
                m_val = np.mean(profile_data[circ][profile_edge_idxs[circ][idx-1]:profile_edge_idxs[circ][idx]])
                tmp_bins.append(bin_circle(profile_edge_idxs[circ][idx-1], profile_edge_idxs[circ][idx], m_val, [rad,rad], len(circ1_data)))
        if ((profile_edge_idxs[circ][0]+len(profile_edges[circ])) - profile_edge_idxs[circ][-1]) > 1:
            m_val = (np.mean(profile_data[circ][profile_edge_idxs[circ][-1]:-1])+np.mean(profile_data[circ][0:profile_edge_idxs[circ][0]]))/2
            tmp_bins.append(bin_circle(profile_edge_idxs[circ][-1], profile_edge_idxs[circ][0]+len(profile_edges[circ]), m_val, [rad,rad], len(circ1_data)))
        #remove bins that are too large or too small
        tmp_bins = [b for b in tmp_bins if (b.get_rad() > 2.5 and b.get_rad() < 14)]
        # Set the angles for the circelbins
        for cbin in tmp_bins:
            centercoords=[profile_coordinates[circ][0][int(cbin.get_center())],profile_coordinates[circ][1][int(cbin.get_center())]]
            cbin.set_angle(centercoords[1], centercoords[0])
        #add bins to list
        profile_bins.append(tmp_bins)
    
    
    circle_groups = []
    for c1 in profile_bins[0]:
        group = [c1]
        center = c1.get_center()
        r = c1.get_rad()
        for c2 in profile_bins[1]:
            if (np.abs(c2.get_center()-center*radius2/radius1) < 8 and c2.get_rad() - r < 3):
                group.append(c2)
                break 
        for c3 in profile_bins[2]:
            if (np.abs(c3.get_center()-center*radius3/radius1) < 8 and c3.get_rad() - r < 3):
                group.append(c3)
                break
        circle_groups.append(group)
    
    
    #sort circle groups using the radius
    circle_groups = [group for group in circle_groups if len(group) == 3]
    circle_groups = sorted(circle_groups, key=lambda x: x[0].get_rad(), reverse=True)
    
    #for plotting 
    circles = []
    colors = sns.color_palette()
    idx = 0
    for group in circle_groups:
        for circ in range(3):
            c = int(group[circ].get_center())
            if c >= len(profile_edges[circ]):
                c -= len(profile_edges[circ])
            r = group[circ].get_rad()
            circ = Circle((profile_coordinates[circ][0][c],profile_coordinates[circ][1][c]) , radius = r, fill = False, ec= colors[idx])
            circles.append(circ)
        idx += 1
        
    #should count consecutive spokes.
    
    #for spoke in range(1,10):
        # start with the biggest found circle from inner
        # spoke is resolved when:
        #   circle size (get_rad()) is comparable between inner/middle/outer (max 2?)
        #   AND the angle (get_angle()) is comparable (max 0.1 rad?)
        # move to next spoke
        #   angle increase? 
    
    count_spokes = len(circle_groups)
        
    fig, axs = plt.subplots(2,3)
    axs[0,0].imshow(image_data_hr,vmin = np.max(image_data)/1.2, vmax=np.max(image_data),cmap=plt.get_cmap("Greys_r"))
    axs[0,0].scatter(rad, rad)
    for circ in range(3):
        axs[0,0].scatter(profile_coordinates[circ][0],profile_coordinates[circ][1],s=1)
    axs[0,0].set_title('Original image')
    
    axs[0,1].imshow(edges)
    axs[0,1].scatter(profile_coordinates[0][0],profile_coordinates[0][1],s=1,color='tab:orange')
    axs[0,1].scatter(profile_coordinates[1][0],profile_coordinates[1][1],s=1,color='g')
    axs[0,1].scatter(profile_coordinates[2][0],profile_coordinates[2][1],s=1,color='r')
    axs[0,1].set_title('Edges in original image')
    
    axs[0,2].imshow(image_data_hr,vmin = np.max(image_data)/1.2, vmax=np.max(image_data),cmap=plt.get_cmap("Greys_r"))
    for circ in circles:    
        axs[0,2].add_patch(circ)
    axs[0,2].set_title('Found spokes ' + str(count_spokes))
    
    
    axs[1,0].plot(profile_edges[0],color='tab:orange')
    axs[1,0].plot(profile_data[0])
    axs[1,0].set_title('Signal & edges inner ring')
    
    axs[1,1].plot(profile_edges[1],color='g')
    axs[1,1].plot(profile_data[1])
    axs[1,1].set_title('Signal & edges middle ring')
    
    axs[1,2].plot(profile_edges[2],color='r')
    axs[1,2].plot(profile_data[2])
    axs[1,2].set_title('Signal & edges outer ring')
    
    plt.show()
    
    return count_spokes, fig


def get_slice_position_error(image_data,x_center_px,y_center_px,axs,title):
    """
    Detect edges of input image
    Using offsets determine the location of the wedges
    Report difference between adjacent wedges
    """
    slice_offsets = [[58,0],[58,-4]]
    searchrange = [10,3]
    
    edges = detect_edges(image_data)
    edges_wedge1 = edges[y_center_px-slice_offsets[0][0]-searchrange[0]:y_center_px-slice_offsets[0][0],
                         x_center_px-slice_offsets[0][1]-searchrange[1]:x_center_px-slice_offsets[0][1]]   
    edges_wedge2 = edges[y_center_px-slice_offsets[1][0]-searchrange[0]:y_center_px-slice_offsets[1][0],
                         x_center_px-slice_offsets[1][1]-searchrange[1]:x_center_px-slice_offsets[1][1]]  
    avg_ind_edge1 = np.mean(np.argwhere(edges_wedge1)[:,0])
    avg_ind_edge2 = np.mean(np.argwhere(edges_wedge2)[:,0])
    slice_pos_error = avg_ind_edge2 - avg_ind_edge1
    
    # Show the resolution insert:
    slice_pos_size = np.array([50,40])
    slice_pos_coordoffsets = np.array([100,19])
    
    image_slice = image_data[y_center_px-slice_pos_coordoffsets[0]:y_center_px-slice_pos_coordoffsets[0]+slice_pos_size[0],
                             x_center_px-slice_pos_coordoffsets[1]:x_center_px-slice_pos_coordoffsets[1]+slice_pos_size[1] ]
    
    y1 = y_center_px-slice_offsets[0][0]-searchrange[0]+avg_ind_edge1 - (y_center_px-slice_pos_coordoffsets[0])
    y2 = y_center_px-slice_offsets[1][0]-searchrange[0]+avg_ind_edge2 - (y_center_px-slice_pos_coordoffsets[0])
    axs.imshow(image_slice,cmap='gray')
    axs.axhline(y=y1,xmin=0.25,xmax=0.5,color='r')
    axs.axhline(y=y2,xmin=0.5,xmax=0.75,color='r')
    axs.set_title(title)
    axs.axis("off")
        
    return slice_pos_error