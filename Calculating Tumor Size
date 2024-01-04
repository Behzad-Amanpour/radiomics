from SimpleITK import GetImageFromArray
from radiomics.featureextractor import RadiomicsFeatureExtractor

# Single DICOM
from pydicom import dcmread
from cv2 import imread, COLOR_BGR2GRAY, cvtColor     #cv2.imread
# import cv2

file = dcmread('D:\\image1.dcm')
Im = file.pixel_array

name = 'D:\\Mask1.png'
mask = cvtColor( imread( name ), COLOR_BGR2GRAY ) # cv2.imread  cv2.cvtColor
mask[mask!=0] = 1

extractor = RadiomicsFeatureExtractor()
extractor.disableAllFeatures() 
extractor.enableFeatureClassByName('shape2D')

f = extractor.execute( GetImageFromArray(Im), GetImageFromArray(mask) )
Maximum_Diameter = f['original_shape2D_MaximumDiameter']

# Pixel Spacing (MIGHT BE MISSED) =============== Behzad Amanpour =============
file.PixelSpacing[0]
file.PixelSpacing[1]

from scipy.ndimage import zoom
Im_resampled = zoom(Im, (file.PixelSpacing[1], file.PixelSpacing[0]) )

import matplotlib.pyplot as plt
plt.imshow(Im, cmap='gray')
plt.imshow(Im_resampled, cmap='gray')

mask_resampled = zoom(mask, (file.PixelSpacing[1], file.PixelSpacing[0]) )

f = extractor.execute( GetImageFromArray(Im_resampled), GetImageFromArray(mask_resampled))
Maximum_Diameter2 = f['original_shape2D_MaximumDiameter']

# Multiple DICOMs ===================== Behzad Amanpour =======================
from os import listdir      # import os  >>   os.listdir()
from os.path import join    # os.path.join
from cv2 import imread, COLOR_BGR2GRAY, cvtColor
from numpy import zeros, vstack    # import numpy as np  >>  np.zeros
from SimpleITK import GetImageFromArray
from radiomics.featureextractor import RadiomicsFeatureExtractor
from scipy.ndimage import zoom

extractor = RadiomicsFeatureExtractor()
extractor.disableAllFeatures() 
extractor.enableFeatureClassByName('shape2D')

path ='D:\\DICOM folder'
files = listdir(path)

Maximum_Diameter = zeros(1)  # np.zeros()

for i in range(2):    # i = 0
    Im_path = join( path, files[i] )   # os.path.join
        
    file = dcmread( Im_path )
    Im = file.pixel_array
    Im = zoom(Im, (file.PixelSpacing[1], file.PixelSpacing[0]) )
    
    mask_path = join( path, files[i+2] )
    mask = cvtColor( imread( mask_path ), COLOR_BGR2GRAY )
    mask[mask!=0] = 1
    mask = zoom(mask, (file.PixelSpacing[1], file.PixelSpacing[0]) )
    
    f = extractor.execute( GetImageFromArray(Im), GetImageFromArray(mask))
    Maximum_Diameter = vstack(( Maximum_Diameter, f['original_shape2D_MaximumDiameter']))

Maximum_Diameter = Maximum_Diameter[1:]
        
        
# Multiple NIfTIs ===================== Behzad Amanpour =======================        
from nibabel import load
from os import listdir
from os.path import join 
from numpy import zeros, vstack, absolute
from SimpleITK import GetImageFromArray
from radiomics.featureextractor import RadiomicsFeatureExtractor
from scipy.ndimage import zoom

extractor = RadiomicsFeatureExtractor()
extractor.disableAllFeatures() 
extractor.enableFeatureClassByName('shape')

path ='D:\\NIfTI folder'
files = listdir(path)

Maximum_Diameter = zeros(1)

for i in range(2):    # i = 0
    Im_path = join( path, files[i] )
    
    file = load( Im_path) 
    Im = file.get_fdata()
    scale = file.affine
        
    Im = zoom(Im, ( absolute(scale[0,0]), absolute(scale[1,1]), absolute(scale[2,2]) ) )
    
    mask_path = join( path, files[i+2] )
    mask = load( mask_path ).get_fdata()
    mask = zoom(mask, ( absolute(scale[0,0]), absolute(scale[1,1]), absolute(scale[2,2]) ) )
    mask[mask<=0.1] = 0
    mask[mask>0.1] = 1
    # a = mask [ 0 < mask ]
    
    f = extractor.execute( GetImageFromArray(Im), GetImageFromArray(mask))
    Maximum_Diameter = vstack(( Maximum_Diameter, f['original_shape_Maximum3DDiameter']))

Maximum_Diameter = Maximum_Diameter[1:]

# Save to drive as Excel ================== Behzad Amanpour ===================
from pandas import DataFrame

name ='D:\\Tumor_size.xlsx'

DataFrame( data = Maximum_Diameter, columns = ['Maximum Diameter']).to_excel(name)
