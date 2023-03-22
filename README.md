# MR_ACR

## Summary
This module performs analysis of measurements of the ACR phantom. Implementation according to:
https://www.acraccreditation.org/-/media/ACRAccreditation/Documents/MRI/LargePhantomGuidance.pdf

## Status
Initial version released (20220630). Some cleanup is still needed.

## Dependencies
pip:
- numpy
- pydicom
- matplotlib
- scipy
- skimage
- seaborn
- cv2 (opencv-contrib-python)

## Acquisition protocol
Use the recommended ACR protocol. (maybe include the Philips examcardsin this git?)

## Selector
A selector for this module should run on dcm_study level as it takes data from the 3 different scans (localizer, T1 and T2). Usually 2 selection rules are sufficient: StationName and PatientName (contains 'acr').

### Input data
dcm_study

### Config
Several config files are provided for different machines:
- mr_acr_philips_sim
- mr_acr_philips_sim30T
- mr_acr_philips_mrl

There are only minor differences: for MRLinac systems skip the Low Contrast Resolution test due to too low SNR.
The config files describe the actions of the module, covering the different tests for the ACR phantom.

### Meta
Limits for the results are taken from the ACR pdf. Differentiate based on field strength.

### Rules
StationName, PatientName

## Analysis
Implementation is based on the recommendations from the ACR
- geometry_z: edge detection and contour extraction
- geometry_xy: edge detection and contour extraction
- resolution: take profile through the grids and peak detection
- slice thickness: find fwhm of ramps
- slice position: edge detection of ramps
- image intensity uniformity: iterative roi search and analysis
- percent signal ghosting: roi analysis
- low contrast object detectability: take circular profiles through expected disk locations and peak detection after fourier filtering

## Results
- Values according to ACR recommendations
- Figures showing calculated metrics/relevant rois
