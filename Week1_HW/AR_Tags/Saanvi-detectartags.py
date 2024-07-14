import cv2

# LOAD IMAGE
tags = cv2.imread('data/two_tags_APRILTAG_16H5.png')

# LOAD DICTIONARY AND PARAMETERS
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# DETECT TAGS IN IMAGE
markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(cv2.cvtColor(tags, cv2.COLOR_BGR2GRAY))

# DRAW DETECTION AND SAVE FILE
cv2.aruco.drawDetectedMarkers(tags, markerCorners, markerIds, borderColor=(255, 0, 0))
cv2.imwrite('detection_two_tags_APRILTAG_16H5.png', tags)

tag1= markerCorners[0]
tag1_inner = tag1[0]
tag1_top_right_corner = tag1_inner[1]
corner_1_x = tag1_top_right_corner[0]

tag2 = markerCorners[1]
tag2_inner = tag2[0]
tag2_bottom_left_corner = tag2_inner[3]
corner_2_x = tag2_bottom_left_corner[0]

pixel_density = (tag1_inner[1] - tag1_inner[0]) / 3
dist_between_tags = (corner_2_x - corner_1_x) / pixel_density

print(markerCorners)
print('tag1:', tag1)
print('tag2:', tag2)
print('c1:', tag1_top_right_corner)
print('c2:', tag2_bottom_left_corner)
print(dist_between_tags)
