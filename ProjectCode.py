import pickle
import cv2
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed
#from moviepy.editor import VideoFileClip
from IPython.display import HTML

nx = 9 #TODO: enter the number of inside corners in x
ny = 6 #TODO: enter the number of inside corners in y

objpoints=[] #3D points in real worls space
imgpoints=[] #2D points in image plane

#prepare object points
objp = np.zeros((6*9,3), np.float32) #3 because of 3 coumns x,y,z, z=0
objp[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)  #x,y coordinates- shape into tho columns

# Read in multiple images
images = glob.glob('./camera_cal/calibration*.jpg')

for fname in images:
    img =  cv2.imread(fname)
    #img = cv2.imread(fname)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        #If corners are found, ass object points, image points  
        imgpoints.append(corners)
        objpoints.append(objp)
        #Drawing detected corners on an image:
        img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
       
# Test undistortion on an image
img = cv2.imread('./camera_cal/calibration15.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
dst = cv2.undistort(img, mtx, dist, None, mtx)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "calibration.p", "wb" ) )

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#cv2.imwrite(os.path.join('./output_images/' + 'undistortion.jpg',dst)

# Choose an image from which to build and demonstrate each step of the pipeline
img = cv2.imread('./test_images/test2.jpg')

hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
s_channel = hls[:,:,2]

# Grayscale image
# NOTE: we already saw that standard grayscaling lost color information for the lane lines
# Explore gradients in other colors spaces / color channels to see what might work better
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Sobel x
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

# Threshold x gradient
thresh_min = 30
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

# Threshold color channel
s_thresh_min = 170
s_thresh_max = 255
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

# Stack each channel to view their individual contributions in green and blue respectively
# This returns a stack of the two binary images, whose components you can see as different colors
color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

# Combine the two binary thresholds
combined_binary = np.zeros_like(sxbinary)
combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

# Plotting thresholded images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Stacked thresholds')
ax1.imshow(color_binary)

ax2.set_title('Combined S channel and gradient thresholds')
ax2.imshow(combined_binary, cmap='gray')
plt.show()
#cv2.imwrite(os.path.join('./output_images/' + 'threshold.jpg')

# undistort image using camera calibration matrix from above
def undistort(combined_binary):
    undist = cv2.undistort(combined_binary, mtx, dist, None, mtx)
    return undist

combined_binary_undistort = undistort(combined_binary)

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.05)
ax1.imshow(combined_binary, cmap='gray')
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(combined_binary_undistort, cmap='gray')
ax2.set_title('Undistorted Image', fontsize=30)

#cv2.imwrite(os.path.join('./output_images/' + 'undistortion_combined_binary.jpg',combined_binary_undistort)
#plt.imshow

#Bird’s-Eye View: Perspective Transformation
h,w = combined_binary_undistort.shape[:2]

#identify four (4) source coordinates points for the perspective transform
# def source and destination
src = np.float32(
    [[575,464],
     [727,464], 
     [258,682], 
     [1100,682]])

dst = np.float32(
    [[450,0],
     [w-450,0], 
     [450,h], 
     [w-450,h]])

def warp(combined_binary_undistort, src, dst):    
    h,w = combined_binary_undistort.shape[:2]
    #Compute the perspective transform, M, given source (src) and destination (dst) points:
    M = cv2.getPerspectiveTransform(src, dst)
    #Compute the inverse perspective transform:
    Minv = cv2.getPerspectiveTransform(dst, src)
    #Warp an image using the perspective transform, M: warp your image to a top-down view
    warped = cv2.warpPerspective(combined_binary_undistort, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv

#apply wrap to or test image
combined_binary_warp, M, Minv = warp(combined_binary_undistort, src, dst)

# Visualize transformation on test image

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.05)
#x = [src[0][0],src[2][0],src[3][0],src[1][0],src[0][0]]
#y = [src[0][1],src[2][1],src[3][1],src[1][1],src[0][1]]
#ax1.plot(x, y, color='#33cc99', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
#ax1.set_ylim([h,0])
#ax1.set_xlim([0,w])
ax1.imshow(combined_binary_undistort, cmap='gray')
ax1.set_title('Original Image with lines', fontsize=30)
ax2.imshow(combined_binary_warp,cmap='gray')
ax2.set_title('Top down, wraped result', fontsize=30)
plt.show()
# Save the result: combined_binary_warp image
cv2.imwrite(os.path.join('./output_images/' + 'test4.jpg'), combined_binary_warp)

#Locate the lane lines
#define image used
#binary_warped = np.uint8(255*combined_binary_warp)
