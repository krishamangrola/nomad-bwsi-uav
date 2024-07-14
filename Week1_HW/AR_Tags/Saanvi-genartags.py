import numpy as np
import cv2
# import matplotlib.pyplot as plt

# LOAD CORRECT TAG DICTIONARY
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
SIZE = 500 # pixels

# CREATE ARRAY FOR MARKER
marker = np.zeros((SIZE, SIZE, 1), dtype=np.uint8)

# DRAW AND SAVE MARKER
IDs = 7, 18, 23

for ID in IDs:
    cv2.aruco.generateImageMarker(arucoDict, ID, SIZE, marker, 1)
    cv2.imwrite('DICT_APRILTAG_16H5_id_{}_{}.png'.format(ID, SIZE), marker)


# plt.imshow('DICT_APRILTAG_16H5_id_7_500.png')