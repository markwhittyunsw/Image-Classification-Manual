# Input: Folder of images
# Output: csv file listing a manual number corresponding to some manual classification from user input
# Mark Whitty
# UNSW
# 20200622 V4 Now recognises image files with capitalized file extensions
# 20190502 V3 Removing bunch architecture specific items
# 20190416 V2 includes button for rotation and window for instructions at startup
# v1 20190411 Starting
# Derived from barcode-QRcodeScannerPy_v8.py 20190409

from __future__ import print_function
import math  # For handling degrees and radians

#from shapely.geometry import Polygon  # Removed as Shapely does't properly compile with pyinstaller.
import os  # For PATH etc.  https://docs.python.org/2/library/os.html
import glob  # For Unix style finding pathnames matching a pattern (like regexp)
import shutil  # For file copying
import numpy as np
import cv2
import time
from datetime import datetime
import sys
#import pdb
from PIL import Image  # For image rotation
from numpy import array
import msvcrt as m  # For keyboard input in Windows only see here:
# https://stackoverflow.com/questions/983354/how-do-i-make-python-to-wait-for-a-pressed-key
import textwrap
import re

# Check whether any file exists in a given directory
# https://stackoverflow.com/questions/33463325/python-check-if-any-file-exists-in-a-given-directory
def does_file_exist_in_dir(path):
    return any(os.path.isfile(os.path.join(path, i)) for i in os.listdir(path))

# Switch error reason based on human input
def switch_class(key):
    switcher = {
        '0': "Class 0",
        '1': "Class 1",
        '2': "Class 2",
        '3': "Class 3",
        '4': "Class 4",
        '5': "Class 5",
    }
    return switcher.get(key, "Invalid key")

# Rotate an image about its centre by a given number of degrees,
# not handling scaling and cropping
def maw_rotate_image(im, angle):
    #(h, w) = im.shape[:2]
    #centre = (w / 2, h / 2)
    #M = cv2.getRotationMatrix2D(centre, angle, 1.0)
    #im_rotated = cv2.warpAffine(im, M, (w, h))
    im2 = Image.fromarray(im)
    im_rotated = im2.rotate(angle, expand=True)
    im_rotated = array(im_rotated)
    return im_rotated

# Resize image according to given maximum height or width
def resize_max(im, max_size):
    height, width = im.shape[:2]
    # only shrink if img is bigger than required
    if max_size < height or max_size < width:
        # get scaling factor
        scaling_factor = max_size / float(height)
        if max_size / float(width) < scaling_factor:
            scaling_factor = max_size / float(width)
        # resize image
        resized_im = cv2.resize(im, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    else:
        resized_im = im
    return resized_im



    # Display results
    #cv2.namedWindow("Results", cv2.WINDOW_AUTOSIZE)
    #smaller_imager = cv2.resize(im, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    #cv2.imshow("Results", smaller_imager)
    #cv2.waitKey(0)
# Borderless windows: https://stackoverflow.com/questions/6512094/how-to-display-an-image-in-full-screen-borderless-window-in-opencv

# Main
if __name__ == '__main__':


    # Show help window to start
    winname = os.path.basename(__file__)
    #"Getting started with image-classifier-manual-keyboard-v2"
    cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)  # Create a named window
    cv2.moveWindow(winname, 100, 100)  # Move it into position
    linewidth = 80  # chars long
    init_window_text = []
    init_window_text.extend(textwrap.wrap(winname[:-3] + " by Mark Whitty (UNSW) on 20190502.", linewidth))
    init_window_text.extend(textwrap.wrap("Usage: Drag a folder containing image files onto the exe, or specify it as a command line argument.", linewidth))
    init_window_text.extend(textwrap.wrap("Keyboard controls:", linewidth))
    init_window_text.extend(textwrap.wrap("  r to rotate this and following images (does not alter the file)", linewidth))
    init_window_text.extend(textwrap.wrap("  q to quit and produce the output file", linewidth))
    init_window_text.extend(textwrap.wrap("  <space> to skip image and not record a class", linewidth))
    init_window_text.extend(textwrap.wrap("  0 - 5 classification levels, ranging from class 0 to class 5", linewidth))
    init_window_text.extend(textwrap.wrap("The output is saved as a tab-separated file named manual_classification.txt in the folder containing the images", linewidth))
    init_window_text.extend(textwrap.wrap("Press any key to continue and display the first image...", linewidth))
    im = Image.fromarray(255*np.ones((500, 800), dtype=np.uint8))
    im = array(im, dtype=np.uint8)

    for il, line in enumerate(init_window_text):
        cv2.putText(im, line, (25, il * 25 + 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0))
    cv2.imshow(winname, im)
    cv2.waitKey()
    # Read destination path
    if len(sys.argv) < 2:
        file_input_path = "."
    else:
        file_input_path = sys.argv[1]

    #file_output_path = os.path.join(file_input_path, os.path.basename(os.path.abspath(file_input_path)) + "_decoded_blanked/")
    max_image_dimension = 1600  # Maximum image dimension, if greater than this will be resized before processing (output image is also resized)

    # Check if input directory exists and contains files
    if not os.path.exists(file_input_path):
        print("Input directory ", file_input_path, " does not exist")
        exit(1)

    if not does_file_exist_in_dir(file_input_path):
        print("Warning: No files in input directory: ", file_input_path,)
        exit(1)

    # Read all files in directory that are in image format as allowed by OpenCV
    # https: // docs.opencv.org / 3.0 - beta / modules / imgcodecs / doc / reading_and_writing_images.html
    print(os.path.abspath(file_input_path))
    input_files = [f for f in os.listdir(file_input_path) if re.search(r'.*\.(jpg|png|bmp|dib|jpe|jpeg|jp2|tif|tiff|JPG|PNG|BMP|DIB|JPE|JPEG|JP2|TIF|TIFF)$', f)]
    input_files = list(map(lambda x: os.path.join(file_input_path, x), input_files))
    num_input_files = len(input_files)

    if num_input_files < 1:
        print("Warning: No image files in input directory: ", file_input_path)
        exit(0)

    # # Create target directory & all intermediate directories if don't exists
    # if not os.path.exists(file_output_path):
    #     os.makedirs(file_output_path)
    #     print("Output directory ", file_output_path, " created ")
    # else:
    #     print("Warning: directory ", file_output_path, " already exists, existing files may be overwritten")


    log_file = open(os.path.join(file_input_path, "manual_classification.txt"), "w+")
    log_file.write("Image number\tInput filename\tClass\tClass label\n")

    ROTATE_STATUS = 0

    print(str(num_input_files), " image files in ", file_input_path, " directory")
    for index, infile in enumerate(input_files, start=1):

        print("Loading image ", index, " of ", num_input_files, " [", int(float(index)/float(num_input_files)*100), "%]")
        cv2.destroyAllWindows()

        # Read image
        im = cv2.imread(infile)

        if(index > 10000):
            print("Warning: Halting execution as more than 10,000 files input")
            break
        # Rotate the image if required (only works for 180 degrees at present)
        #im = maw_rotate_image(im, 180)
        im = resize_max(im, 800)

        winname = "Image " + str(os.path.basename(infile))
        cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)  # Create a named window
        cv2.moveWindow(winname, 100, 100)  # Move it into position
        rotated_im = maw_rotate_image(im, ROTATE_STATUS)
        cv2.imshow(winname, rotated_im)
        key = chr(cv2.waitKey())
        while key == 'r':
            ROTATE_STATUS = (ROTATE_STATUS + 90) % 360
            rotated_im  = maw_rotate_image(im, ROTATE_STATUS)
            cv2.imshow(winname, rotated_im)
            key = chr(cv2.waitKey())
        if key == 'q':
            log_file.close()
            cv2.destroyAllWindows()
            exit(0)
        if key == ' ':
            log_file.write(str(index) + "\t" + os.path.basename(infile) + "\n")
            continue
        if switch_class(key) == "Invalid key":
            print("Invalid key: " + str(key))
            log_file.write(str(index) + "\t" + os.path.basename(infile) + "\n")
            continue
        log_file.write(str(index) + "\t" + os.path.basename(infile) + "\t" + key + "\t" + str(switch_class(key)) + "\n")

    log_file.close()
