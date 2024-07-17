import numpy as np
import cv2
import glob

# Define checkerboard_dims as (8, 6)
checkerboard_dims = (8,6)

#Define number of corners
corners = checkerboard_dims[0] * checkerboard_dims[1]

# Create objp as a zero array of shape (number of corners, 3), float32
objp = np.zeros((corners, 3), dtype=np.float32)
# Set the first two columns of objp to the coordinate grid of corners
objp[:,:2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)

# Initialize objpoints as an empty list
objpoints = []
# Initialize imgpoints as an empty list
imgpoints = []

# Load all checkerboard images using glob ('path/to/images/*.jpg')
imgs = [img for img in glob.iglob('C:/Users/owent/Documents/BWSI-UAV/camera_calibration_photo_mosaic/calibration_photos/*.jpg')]

# For each image in images:
for fname in imgs:
   
    # Read the image & convert to grayscale
    img = cv2.imread(fname)
    if img is not None:
        print("Image Loaded")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Look for chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None)

    # If corners are found:
    if ret:
        # Append objp to objpoints
        objpoints.append(objp)

        # Refine corner positions using cornerSubPix
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        # Append refined corners to imgpoints
        imgpoints.append(corners2)
        
        # Optionally, draw chessboard corners on the image
        # cv2.drawChessboardCorners(img, checkerboard_dims, corners2, ret)

        # Optionally, display the image with drawn corners
        # cv2.imshow('img', img)
        # cv2.waitKey(10000)
    
# Destroy all OpenCV windows
cv2.destroyAllWindows()

# Calibrate the camera using calibrateCamera with objpoints, imgpoints, and image size
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the calibration results (camera matrix, distortion coefficients) to a file. 
np.savez('camera_calibration.npz', mtx=mtx, dist=dist)

# Verify the calibration:
# Initialize mean_error to 0
mean_error = 0

# For each pair of object points and image points:
for i in range(len(objpoints)):

    # Project the object points to image points using projectPoints
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    
    # Compute the error between the projected and actual image points
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    
    # Accumulate the error
    mean_error += error

# Print the total average error
print("Total average error: {}".format(mean_error/len(objpoints)))