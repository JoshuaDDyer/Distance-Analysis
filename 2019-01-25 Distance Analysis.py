"""sample_script_annotated.py

This is a quick sample script for Josh
to see how "easy" doing this kind of
object based colocalisation can be in Python

NOTE:
I would usually lay out the code differently, in a
more "modular" fashion, such that each discreet piece of
functionality is in it's own function.

However, for simple demonstrative purposes, we'll start
with a very basic script.

This "triple quoted" string is how you can write
multi-line strings in python, and an initial
triple quoted string like this one, is used by python
to generate documentation for this script.

There is also an unannotated version available!

J. Metz <metz.jp@gmail.com>
"""
input_folder_name='data' #input folder name
output_folder='tables' #output folder name

import numpy as np

# Normal single-line comments are created using the hash symbol (#)
# To start with, we will import modules that we will later need.
# This set of import statements can of course be added to as the
# script is developed, but it is recommended practice to put the
# imports at the top of the script.

import czifile

# The czifile module is used to read czi files
# it is available from the Python Package Index, pypi
# and installable from the command-line using:
#
#     pip install czifile
#
# The same syntax is used for installing any module from the
# python package index (online repository of python modules).

import matplotlib.pyplot as plt

# Matplotlib is probably the most common and widely used plotting
# module for Python.
# A simple plotting interface is provided by it's pyplot submodule
# which we import using the "dot" notation above.
# Lastly, the "as" allows us to use an alias for the module, so
# instead of writing
#
#     matplotlib.pyplot.plot(...
#
# we can instead write
#
#    plt.plot(...
#

# Next come the image processing libraries...

import scipy.ndimage as ndi
import skimage.filters as skfilt

# And lastly we're going to write a csv table
# so we'll use the Python-standard-library module, csv

import csv

# Now I'm going to create a variable called filename, which
# I will use to hold the path (as a string) to the data file
# we're going to work on

# NOTE: In python you can use single or double quotes for strings
# i.e. anything between a set of single or double quotes (but not mixed!)
# is a string.
# A string is short for a string of characters, and it is one of the
# most basic datatypes in python, along with numbers.

# Special note about this path format
# I've used a "relative" path format, which means that Python is
# going to try and find the file 12-10-18_ACTUAL_1hr_Con_1-Scene-1.czi
# in a folder called data, which should be located in the folder that we
# run this script from.
import os


# ------------------------------
# Add in calibration of pixels in xy and z directions
# ------------------------------
zmicsperpix = 0.75
xymicsperpix = 0.08 


li=os.listdir(input_folder_name)
filename_list=[]
for el in li:
    if el[-3:]=='czi':
	    filename_list.append(el)
        
print(filename_list)

#for filename in list_of_files:
#filename = 'data/12-10-18 ACTUAL 24hr Con 3-Scene-1.czi'
for i,filename in enumerate(filename_list):
# Next we're going to load the data using the czifile module we imported...
    
    print('processing file{}, this is file {} of {}'.format(filename,i+1,len(filename_list)))
    filepath = 'data/{}'.format(filename)
    mydata = czifile.imread(filepath)

# Here we loaded the data and assigned it to a new variable called mydata
# NB: variable names are up to us! I tend to try to stick to "sensible"
# variable names that will help me understand what the variable refers to,
# but had I wanted to, I could have called the variable "banana", or "ds2832"
# There are some rules about variable names, such as not using a function name,
# and they have to start with a letter, but more on that as you learn more
# python.

# Now because of the way czifile works (I found this out while writing
# this script!) the data is loaded with more dimensions than we actually need.
# To see this, we can "print" (i.e. write) the shape attribute of mydata to
# the terminal:

    print("The data we loaded has shape")
    print(mydata.shape)
#
# This should show you a shape of 1x1x1x1x2x20x1040x1040x1, i.e lots of
# pointless size 1 dimensions.
# The fact that we can use mydata.shape, i.e. that this object has a shape
# attribute # is because it's probably the most important non-built-in
# data-type in python: a Numpy array.
#
# Numpy is a great module for numerical computing, and lots of it's strength is
# encapsulated in the Numpy array data type, which is used to represent
# multi-dimensional data (and often used for images, stacks, etc).
# Anyway, we can "squeeze" away those unnecessary dimensions, using the squeeze
# "member-function".
# NOTE: A member function, like an attribute, is something that "comes with" an
# object, and can be accessed using the object-dot notation like an attribute.

    mydata = mydata.squeeze()

# Here we called the squeeze function by using the open-close parentheses "()"
# and re-assigned the output of the function (a squeezed version of mydata)
# back into mydata. Let's confirm that the data is now more sensible:

    print("The data we squeezed has shape")
    print(mydata.shape)

# Now you should see the shape come up as: (2, 20, 1040, 1040)
# which is much more sensible; there are 2 channels, 20 z-slices,
# and each image is 1040 x 1040!
# But of course this is represented now as just a 4-dimensional array
# in python!
# As we want to work with the channels independently, we will assign
# each of the channels into a new variable...

    channel1 = mydata[0]
    channel2 = mydata[1]

# Here we've used "slicing" to access sub-arrays of the array.
# Slicing is done by using square-brackets, and then the slice information.
# Slicing can get quite complicated, but as we just needed to access the
# components of the first dimension, we could get away with a really simple
# slicing operation!
# Note: I've used 0 to access the first channel, and 1 to access the second,
# because like most programming languages, indexing starts with 0!

# Let's confirm out channel data shapes (just so we're sure...)

    print("Channel 1 data has shape:", channel1.shape)
    print("Channel 2 data has shape:", channel2.shape)

# Note: Here I show that actually print can take multiple
# inputs, separated by commas, and it prints them all to the terminal
# on the same line separated by spaces.

# Great! Now we're good to start actually processing the
# data -> we have 2 3-d arrays, and can start to filter,
# threshold, and distance transform in the same way as we
# did in imagej

# For this we're going to use the image processing library
# - scipy.ndimage - we imported.

# First of all, let's apply a gaussian blur to smooth out the
# single-pixel scale noise...

    filtered1 = ndi.gaussian_filter(channel1, 1.0)
    filtered2 = ndi.gaussian_filter(channel2, 1.0)



# Here I've used a "sigma" 1 gaussian filter, (sigma
# corresponds to the width of the gaussian profile used to
# blur the data.

# Next we need to calculate a sensible threshold.
# In the same way that imagej has the automated threshold
# methods, scikit-image (skimage) has these in it's filters
# submodule;

    threshold_value1 = skfilt.threshold_otsu(filtered1)
    threshold_value2 = skfilt.threshold_otsu(filtered2)

# Note that threshold_otsu just takes any-dimensional data
# and calculates a single threshold from all of it!
# No need to worry about slices etc ;)

# Now we apply this threshold... which as we have an array,
# we can simply apply a "comparison operator"...

    mask1 = filtered1 >= threshold_value1
    mask2 = filtered2 >= threshold_value2
    
# And voila! -> two binary stacks (3d arrays).
# Our final main operation is to distance transform
# one of these... scipy.ndimage has this for us!

# Note: because the distance map here measures how
# far to the nearest OFF pixel (0), we need to
# invert the mask while performing this function

# For logical/boolean masks, this is done using "~"


    # NOTE: Includes pixel sampling factors;
    # so distances will be in microns
    distancemap2 = ndi.distance_transform_edt(
        ~mask2,
        sampling=[zmicsperpix, xymicsperpix, xymicsperpix],
    )

# Aside: the edt part of the function just stands for
# Exact euclidean distance transform, as the module
# also has other ways of calculating different
# distance transforms!

# Ok so we're almost there!
# To verify that it's all looking right, we can
# create image plots of what's going on.
# For that we choose a "slice" as the image has to be
# 2d (for simplicity... 3d plotting functions exist but
# are more tricky!).
# We may as well take a middle-ish slice like 10, of
# this 20-z-plane data.

# Create a new figure object, and set the window title

    size = channel1[6].shape
    plt.figure(
        "Mask 1 (slice 6)",
        figsize=(12, 12*size[0]/size[1]),
        dpi=size[1]/12,
    )

# Now show the corresponding image data, using a "gray"
# colormap

    plt.axes([0,0,1,1])
    plt.imshow(channel1[6], cmap='gray')
    # plt.imshow(np.ma.masked_where(mask1[6]==0, mask1[6]), cmap='hsv', alpha=0.5)
    plt.contour(mask1[6], levels=[0.5], colors=['r'])
    plt.savefig('{}_mask1.png'.format(filepath))
    plt.close()

    # Similarly for a few other images...

    plt.figure(
        "Mask 2 (slice 6)",
        figsize=(12, 12*size[0]/size[1]),
        dpi=size[1]/12,
    )
    plt.axes([0,0,1,1])
    plt.imshow(channel2[6], cmap='gray')
    # plt.imshow(np.ma.masked_where(mask2[6]==0, mask2[6]), cmap='hsv', alpha=0.5)
    plt.contour(mask2[6], levels=[0.5], colors=['r'])
    plt.savefig('{}_mask2.png'.format(filepath))
    plt.close()

    print("Created overlay images", flush=True)

    # plt.figure("Hist 1")
    # plt.hist(channel1[6].flatten(), 256, log=True)
    # plt.figure("Hist 2")
    # plt.hist(channel2[6].flatten(), 256, log=True)

#plt.figure("Distance transform of mask2")
#plt.imshow(distancemap2[10], cmap='jet')
# Note: Here I'm going to use the 'jet' colourmap!

# Once plot objects have been created, we call "show"
# to display them and pause the execution of the code
# until they've all been closed again
# But we won't do this until the end of the script for now.

# So next, for each object in mask1, let's
# See how far it's border pixels are from the
# nearest object in mask2 (using distancemap2)...

# First we create a "labelled" array, which
# performs part of what happens with
# "Analyze particles" in imagej; it identifies
# connected clumps of pixels and gives them all a unique
# label.

    labels1, num_objects1 = ndi.label(mask1)

    if num_objects1>1000:
        print('oh shit loads of stuff, abort this')
        continue
#	continue
# NB: As this function is in scipy.ndimage - it happily handles
# N-d data for us!
# It also outputs the number of objects found, so we can show that
# in the terminal...

    print("Number of objects in mask1:", num_objects1)

# Let's see what that looks like
#plt.figure("Labels of mask1")
#plt.imshow(labels1[10], cmap='jet')

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
    dist_stats_list=[['min','max','mean','sum','total']]

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

# NB: Usually I would use a "context manager" to not have to
# explicitly close the file object, but lets leave that
# for next time ;)

#plt.show()

# Hopefully those plots make sense!
