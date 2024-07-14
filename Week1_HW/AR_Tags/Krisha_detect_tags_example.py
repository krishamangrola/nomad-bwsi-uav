import cv2
import numpy as np

# load in the image 
image = cv2.imread('two_tags_APRILTAG_16H5.png')
tag_size = 3.0  # centimeters

# load dictionary
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# detect tags in image 
markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

# check if 2 tags are detected 
if markerIds is not None and len(markerIds) >= 2 and len(markerCorners) >= 2:
    # extract the center
    center1 = np.mean(markerCorners[0][0], axis=0)
    center2 = np.mean(markerCorners[1][0], axis=0)

    # calc diff between centers (in pixels)
    distance_pixels = np.sqrt(np.sum((center1 - center2) ** 2))

    # conv dist to cm 
    pixel_width = image.shape[1]  # width of the image in pixels
    distance_cm = (distance_pixels / pixel_width) * tag_size

    print(f"The distance between the tags is approximately {distance_cm:.2f} centimeters.")
else:
    print("Less than two markers detected or no markers detected.")
