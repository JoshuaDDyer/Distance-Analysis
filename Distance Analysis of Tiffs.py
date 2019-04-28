input_folder_name='data' #input folder name
output_folder='tables' #output folder name

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.filters as skfilt
import csv
import os


# ------------------------------
# Add in calibration of pixels in xy and z directions
# ------------------------------
zmicsperpix = 0.75
xymicsperpix = 0.08 

#find .tif files in input_folder_name, add to directory
li=os.listdir(input_folder_name)
filename_list=[]
for el in li:
    if el[-3:]=='tif':
	    filename_list.append(el)
print(filename_list)



for i,filename in enumerate(filename_list):

# load the data using tifffile, unlike when loading .czi files, no unnecessary
#dimensions are added, meaning that we do not need to use the .squeeze function. 
    
    print('processing file{}, this is file {} of {}'.format(filename,i+1,len(filename_list)))
    filepath = 'data/{}'.format(filename)
    mydata = tifffile.imread(filepath)
# show dimensions of loaded image
    print("The data we loaded has shape")
    print(mydata.shape)
# Switch axis describing no. z-stacks with axis describing channels - easier slicing
    mydatareshaped = np.moveaxis(mydata,0,1)
    print (mydatareshaped.shape)
#split channels 
    channel1 = mydatareshaped[0]
    channel2 = mydatareshaped[1]
    print("Channel 1 data has shape:", channel1.shape)
    print("Channel 2 data has shape:", channel2.shape)
#at this point extra filtering steps can be inserted; I have pre-filtered tis data in image-j so no need for extra filtering
    filtered1 = channel1
    filtered2 = channel2
#identify thresholding values using threshold_otsu
    threshold_value1 = skfilt.threshold_otsu(filtered1)
    threshold_value2 = skfilt.threshold_otsu(filtered2)
#apply threshold using comparison operator to generate 3D binary stacks
    mask1 = filtered1 >= threshold_value1
    mask2 = filtered2 >= threshold_value2

#Generate distance map of mask2 - Note: because the distance map here measures how
# far to the nearest OFF pixel (0), we need to
# invert the mask while performing this function
# Aside: the edt part of the function just stands for
# Exact euclidean distance transform, as the module
# also has other ways of calculating different
# distance transforms!
# NOTE: Includes pixel sampling factors;
    # so distances will be in microns
    
    distancemap2 = ndi.distance_transform_edt(
        ~mask2,
        sampling=[zmicsperpix, xymicsperpix, xymicsperpix],
    )

# We now create overlay figures for channel 1 and 2 
# and save to the input folder to allow 
# the thresholding to be assessed    
# Create a new figure object, and set the window title

#channel1 overlay images:
    size = channel1[6].shape
    plt.figure(
        "Mask 1 (slice 6)",
        figsize=(12, 12*size[0]/size[1]),
        dpi=size[1]/12,
    )
    plt.axes([0,0,1,1])
    plt.imshow(channel1[6], cmap='gray')
    plt.contour(mask1[6], levels=[0.5], colors=['r'])
    plt.savefig('{}_mask1.png'.format(filepath))
    plt.close()

#channel2 overlay images:
    plt.figure(
        "Mask 2 (slice 6)",
        figsize=(12, 12*size[0]/size[1]),
        dpi=size[1]/12,
    )
    plt.axes([0,0,1,1])
    plt.imshow(channel2[6], cmap='gray')
    plt.contour(mask2[6], levels=[0.5], colors=['r'])
    plt.savefig('{}_mask2.png'.format(filepath))
    plt.close()
    print("Created overlay images", flush=True)

# So next, for each object in mask1, let's
# See how far it's border pixels are from the
# nearest object in mask2 (using distancemap2)...

# First we create a "labelled" array, which
# performs part of what happens with
# "Analyze particles" in imagej; it identifies
# connected clumps of pixels and gives them all a unique
# label.

    labels1, num_objects1 = ndi.label(mask1)

    if num_objects1>500:
        print('oh shit loads of stuff, abort this')
        continue
#	continue
# NB: As this function is in scipy.ndimage - it happily handles
# N-d data for us!
# It also outputs the number of objects found, so we can show that
# in the terminal...

    print("Number of objects in mask1:", num_objects1)


# Next we want to process each labelled region (~object)
# individually, so we will use a for-loop
# We know how many objects there are from num_objects1

# NOTE: The Python range function, when called with two inputs
# generates the numbers input1, input1+1, input1+2, ... , input2-1
# I.e. it does not generate input2, so if we want to have a
# range of numbers go to X, then we have to use X+1 as the second input!

# We're also going to create a list of distances - one
# entry per object, but each entry itself is going to
# be a list of distances!

# First, let's create an empty list using []
    distances_list = []
    dist_stats_list=[['min distance','max distance','mean distance','median distance','sum distance','total number of bacterial perimiter pixels']]

    for label in range(1, num_objects1+1):

    # Now we generate a mask of just this object
    # which is kind of like an ROI object in imagej
    # We can do this by using the comparison operator ==
    # which evaluates to true where values in labels1 are equal
    # to label
        mask_obj = labels1 == label

    # Now we can generate an outline of this region by performing
    # binary erosion of this region and getting the pixels where
    # mask_obj is True, but the eroded version is False!
        eroded = ndi.binary_erosion(mask_obj)
        outline = mask_obj & ~eroded

    # Now we can get all the distances of the outline pixels
    # to the nearest object in mask2!
        distances = distancemap2[outline]
        dist_arr=np.array(distances)
        dist_stats=[
            np.min(dist_arr),
            np.max(dist_arr),
            np.mean(dist_arr),
            np.median(dist_arr),
            np.sum(dist_arr),
            dist_arr.shape[0]
        ]
    # This uses "logical indexing", i.e. we can pass in a mask array
    # to extract only pixels in distancemap2 where the mask array
    # (in this case outline) is True.

    # Lastly we just need to decide what to do with the distances...
    # For now, let's create a table, so we're going to append the
    # current distances to the end of the list
        distances_list.append(distances)
        dist_stats_list.append(dist_stats)

# Now that we have all the distances in a list, we could
# perform statistical tests on them, or get their means,
# maxima etc, but for now let's just write them to a
# CSV (comma-seperated-values) file which can be loaded
# into a variety of software (including Excel...) for
# later inspection!

# To do this we're going to "open" a file (which returns
# a "file-object" representing the open file
# and then use the Python standard library's csv module
# to send the data to the open file!
    file_out = open("{}/distance_table_{}.csv".format(output_folder,filename), "w")
    writer = csv.writer(file_out)
    writer.writerows(distances_list)
    file_out.close()

    file_out = open("{}/stats_table{}.csv".format(output_folder,filename), "w")
    writer = csv.writer(file_out)
    writer.writerows(dist_stats_list)
    file_out.close()
