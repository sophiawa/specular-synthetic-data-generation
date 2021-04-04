import os
import h5py
import argparse
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import sys
import json
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import morphology
import cv2
import imageio

parser = argparse.ArgumentParser("Script to visualize hdf5 files")

parser.add_argument('hdf5_paths', nargs='+', help='Path to hdf5 file/s')
parser.add_argument('--keys', nargs='+', help='Keys that should be visualized. If none is given, all keys are visualized.', default=None)
parser.add_argument('--image_index', nargs='+', help='Image index when generating images')
parser.add_argument('--rgb_keys', nargs='+', help='Keys that should be interpreted as rgb data.', default=["colors", "normals"])
parser.add_argument('--flow_keys', nargs='+', help='Keys that should be interpreted as optical flow data.', default=["forward_flow", "backward_flow"])
parser.add_argument('--segmap_keys', nargs='+', help='Keys that should be interpreted as segmentation data.', default=["segmap"])
parser.add_argument('--segcolormap_keys', nargs='+', help='Keys that point to the segmentation color maps corresponding to the configured segmap_keys.', default=["segcolormap"])
parser.add_argument('--other_non_rgb_keys', nargs='+', help='Keys that contain additional non-RGB data which should be visualized using a jet color map.', default=["distance", "depth"])

args = parser.parse_args()


def vis_data(key, data, full_hdf5_data, file_label, index):
    # If key is valid and does not contain segmentation data, create figure and add title
    if key in args.flow_keys + args.rgb_keys + args.other_non_rgb_keys:
        plt.figure()
        plt.title("{} in {}".format(key, file_label))

    if key in args.flow_keys:
        try:
            # This import here is ugly, but else everytime someone uses this script it demands opencv and the progressbar
            sys.path.append(os.path.join(os.path.dirname(__file__)))
            from utils import flow_to_rgb
        except ImportError:
            raise ImportError("Using .hdf5 containers, which contain flow images needs opencv-python and progressbar "
                              "to be installed!")

        # Visualize optical flow
        #plt.imshow(flow_to_rgb(data), cmap='jet')
        plt.savefig(key + '.png')
        print(key + 'flow')

    elif key in args.other_non_rgb_keys:
        # Make sure the data has only one channel, otherwise matplotlib will treat it as an rgb image
        if len(data.shape) == 3:
            if data.shape[2] != 1:
                print("Warning: The data with key '" + key + "' has more than one channel which would not allow using a jet color map. Therefore only the first channel is visualized.")
            data = data[:, :, 0]
        
        #plt.imshow(data, cmap='jet')
        plt.savefig('examples/coco_annotations/output/' + key + str(index) + '.png')
        

    elif key in args.rgb_keys:
        #plt.imshow(data)
        
        if key == "normals":
            imageio.imwrite('examples/coco_annotations/output/' + key + str(index) + '.exr', data)
        else:
            data = data.astype('uint8')
            im = Image.fromarray(data)
            im.save('examples/coco_annotations/output/' + key + str(index) + '.jpg')
        # normals, colors


def vis_file(path, index):
    # Check if file exists
    if os.path.exists(path):
        if os.path.isfile(path):
            with h5py.File(path, 'r') as data:
                print(path + " contains the following keys: " + str(data.keys()))

                # Select only a subset of keys if args.keys is given
                if args.keys is not None:
                    keys = [key for key in data.keys() if key in args.keys]
                else:
                    keys = [key for key in data.keys()]

                # Visualize every key
                for key in keys:
                    value = np.array(data[key])
                    # Check if it is a stereo image
                    if len(value.shape) >= 3 and value.shape[0] == 2:
                        # Visualize both eyes separately
                        for i, img in enumerate(value):
                            vis_data(key, img, data, os.path.basename(path) + (" (left)" if i == 0 else " (right)"), index)
                    else:
                        vis_data(key, value, data, os.path.basename(path), index)

                    if key == "depth":
                        # Get x-gradient in "sx"
                        sx = ndimage.sobel(value,axis=0,mode='constant')
                        # Get y-gradient in "sy"
                        sy = ndimage.sobel(value,axis=1,mode='constant')
                        # Get square root of sum of squares
                        sobel=np.hypot(sx,sy)
                    

                        
                        sobel[:,0] = 0
                        sobel[:,-1] = 0
                        sobel[0,:] = 0
                        sobel[-1,:] = 0

                        count = len(sobel) * len(sobel[0])
                        i = 3
                        where = np.where(sobel >= i)
                        while len(where[0]) > 0.1 * count:
                            i -= 0.5
                            where = np.where(sobel >= i)

            
                    
                    
                        ok = sobel.astype(int)
                        wheree = np.where(ok != 0)
                        ok = ok.astype('uint8')
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                        kernel = kernel.astype('uint8')
                        sobel = morphology.dilation(sobel)
                        sobel = morphology.dilation(sobel)
                        sobel = morphology.dilation(sobel)
                        sobel = morphology.dilation(sobel)
                        sobel = morphology.dilation(sobel)
                        sobel = morphology.dilation(sobel)
                        sobel = morphology.dilation(sobel)
                        sobel = morphology.dilation(sobel)
                        wheree = np.where(sobel >= 3)
                        sobel[wheree] = 10


                        img = cv2.imread('mask' + str(index) + '.png')
                        mask = np.zeros(img.shape, dtype=np.uint8)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                        cv2.drawContours(mask, cnts, -1, 5, thickness=6)
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                        where1 = np.where(mask == 0)
                        where2 = np.where(mask != 0)
                        sobel = sobel.astype('uint8')
                        test = cv2.bitwise_and(sobel, gray)
                        maskim = Image.fromarray(mask, mode='L')
                        draw = ImageDraw.Draw(maskim)
                        mask1 = np.array(maskim)
                        mask1[where] = 0
                        mask1[:,-3:] = 0
                        mask1[:,:3] = 0
                        sobel[where2] = 10
                        test[where2] = 10
                        mask2 = mask1 * 255
                        mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2RGB)
                        mask2[where] = (72, 209, 204)
                        mask2[where2] = (173, 255, 47)
                        mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
                        mask1 = morphology.binary_erosion(mask1)
                        mask1 = mask1.astype(int)
                        sobel = sobel.astype(int)
                        test = test.astype(int)
                        where4 = np.where(sobel != 0)
                        sobel[where4] = 10
                        try1 = cv2.add(test, mask1)
                        where3 = np.where(mask1 != 0)
                        try1[where3] = 30
                        try1[:,-3:] = 0
                        try1[:,:3] = 0
                        bru1 = np.where(try1 == 1)
                        bru2 = np.where(try1 == 2)
                        bru10 = np.where(try1 == 10)
                        bru30 = np.where(try1 == 30)
                        try1 = try1.astype('float64')
                        try1[bru1] = 0.00392157
                        try1[bru2] = 0.00392157
                        try1[bru10] = 0.00392157
                        try1[bru30] = 0.00784314
                        imageio.imwrite('outlines' + str(index) + '.png', try1)
                    


        else:
            print("The path is not a file")
    else:
        print("The file does not exist: {}".format(args.hdf5))


# Visualize all given files
for path in args.hdf5_paths:
    vis_file(path, args.image_index)
#plt.show()
