# Scroll down for summary

# !pip install pyradiomics
# !pip install SimpleITK
from radiomics.featureextractor import RadiomicsFeatureExtractor
from SimpleITK import GetImageFromArray

# Radomics Setup ------------------- Behzad Amanpour -----------------
extractor = RadiomicsFeatureExtractor()
# extractor = RadiomicsFeatureExtractor( interpolator = 'sitkGaussian', normalize=True )
settings = extractor.settings 
for key in settings:
    print( key, ' : ', extractor.settings[key] )

# Features management ------------------------------------------------
print(extractor.featureClassNames)
print (extractor.enabledFeatures)

extractor.enableFeatureClassByName('shape2D') # default shape is 3D

extractor.disableAllFeatures()
extractor.enableFeatureClassByName('firstorder')
extractor.enableFeatureClassByName('glcm')
extractor.enableFeatureClassByName('shape2D')

# Filtering Management ------------- Behzad Amanpour ----------------

extractor.enableImageTypeByName('Wavelet')
print(extractor.enabledImagetypes)

extractor2 = RadiomicsFeatureExtractor(sigma=[1, 3])  # level=2
extractor2.enableAllImageTypes()
print(extractor2.enabledImagetypes)

# Loading Image ------------------ Behzad Amanpour --------------------
# the image is medical & 2D (dicom format), the mask format in png-----
# ---------------------------------------------------------------------
from pydicom import dcmread
import numpy as np
import matplotlib.pyplot as plt
import cv2
def fig_show( Image1 , Image2 ):
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(Image1, cmap='gray')
    fig.add_subplot(1, 2, 2)
    plt.imshow(Image2, cmap='gray')

file = dcmread('Image1.dcm')
image = file.pixel_array
mask = cv2.imread('Mask1.png')
mask = cv2.cvtColor(cv2.imread('Mask1.png'), cv2.COLOR_BGR2GRAY)
mask[ mask!=0 ] = 1
fig_show (image, mask)

# Feature Extraction --------- Behzad Amanpour --------------------
#from SimpleITK import GetImageFromArray
result = extractor.execute( GetImageFromArray(image), GetImageFromArray(mask) )
result2 = extractor2.execute( GetImageFromArray(image), GetImageFromArray(mask) )

# Making Data Frame ------------- Behzad Amanpour -----------------
import pandas as pd 
result2 = pd.DataFrame( result.items() ) 
result3 = result2.drop( result2.index[0:22] )
array = result3.values
array_val = array[:,1]
# array_val = pd.DataFrame(result.items()).drop(pd.DataFrame(result.items()).index[0:22]).values[:,1]

# Summary for Multiple Images --------------- Behzad Amanpour -----------------
from radiomics.featureextractor import RadiomicsFeatureExtractor
from SimpleITK import GetImageFromArray
from pydicom import dcmread
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
def fig_show( Image1 , Image2 ):
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(Image1, cmap='gray')
    fig.add_subplot(1, 2, 2)
    plt.imshow(Image2, cmap='gray')
extractor = RadiomicsFeatureExtractor() #normalize=True
extractor.enableFeatureClassByName('shape2D')

array_val = np.zeros((2,102))
for i in range(2):
    print(i)
    image_name = 'Image' + str(i+1) + '.dcm'  # the names are Image0.dcm, Image1.dcm, ...
    image = dcmread(image_name).pixel_array
    mask_name = 'Mask' + str(i+1) + '.png'    # the names are Mask0.png, Mask1.png, ...
    mask = cv2.cvtColor(cv2.imread(mask_name), cv2.COLOR_BGR2GRAY)
    mask[ mask!=0 ] = 1
    fig_show (image, mask)
    result = extractor.execute( GetImageFromArray(image), GetImageFromArray(mask) )
    result = pd.DataFrame( result.items() )
    result = result.drop(result.index[0:22])
    array_val[i , :] = result.values[:,1].T
array = result.values
