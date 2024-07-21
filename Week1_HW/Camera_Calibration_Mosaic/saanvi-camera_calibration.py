import numpy as np
import cv2
import glob

# Define checkerboard_dims as (8, 6) 
checkerboard_dims = (8,6)

corners = checkerboard_dims[0] * checkerboard_dims[1]

# Create objp as a zero array of shape (number of corners, 3), float32
objp = np.zeros((corners, 3), np.float32)

# Set the first two columns of objp to the coordinate grid of corners
objp[:,:2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1,2)
print(objp)

# Initialize objpoints as an empty list
objpoints = []
# Initialize imgpoints as an empty list
imgpoints = []

# Load all checkerboard images using glob ('path/to/images/*.jpg')
path = '/home/saanvi/bwsi-uav2/laboratory_2024/week_1_Hw/camera_calibration_photo_mosaic/calibration_photos/*.jpg'
images = [img for img in glob.iglob(path)]

for image in images:
#     Read the image
    img = cv2.imread(image)
#     Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     Find the chessboard corners in the grayscale image
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None)

#     If corners are found:
    if ret is not None:
#         Append objp to objpoints
        objpoints.append(objp)
#         Refine corner positions using cornerSubPix
        criteria = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#         Append refined corners to imgpoints
        imgpoints.append(corners2)

#         Optionally, draw chessboard corners on the image
#         cv2.drawChessboardCorners(img, checkerboard_dims, corners2, ret)
# #         Optionally, display the image with drawn corners
#         cv2.imshow('img', img)
# #         Wait for a short period
#         cv2.waitKey(10000)
    
# Destroy all OpenCV windows
cv2.destroyAllWindows()

# Calibrate the camera using calibrateCamera with objpoints, imgpoints, and image size
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# Get the camera matrix, distortion coefficients, rotation vectors, and translation vectors

# Save the calibration results (camera matrix, distortion coefficients) to a file.
np.savez('camera_calibration.npz', mtx=mtx, dist=dist) 
print(mtx)
print(dist)
# A common and convenient format for storing camera calibration data is the NumPy .npz file format,
#     which allows you to store multiple NumPy arrays in a single compressed file.

# Verify the calibration:
#     Initialize mean_error to 0
mean_error = 0

#     For each pair of object points and image points:
for i in range(len(objpoints)):
#         Project the object points to image points using projectPoints
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#         Compute the error between the projected and actual image points
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
#         Accumulate the error
    mean_error = mean_error + error
#     Compute the average error
#     Print the total average error
print("total average error:", mean_error/len(objpoints))