import cv2
import numpy as np


tags = cv2.imread(r"C:\Users\Steven Li\Downloads\BWSI-UAV-intro_to_ar_tags\BWSI-UAV-intro_to_ar_tags-22a92bf\two_tags_APRILTAG_16H5.png")
cv2.imshow( "tags",tags)
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, parameters)
corners, ids, rejects = detector.detectMarkers(cv2.cvtColor(tags, cv2.COLOR_BGR2GRAY))
detection = cv2.aruco.drawDetectedMarkers(tags, corners, borderColor=(255, 0, 0))
cv2.imwrite('detection_two_tags_ARUCO_APRILTAG_16H5.png', detection)
center1 = np.mean(corners[0][0], axis=0)
center2 = np.mean(corners[1][0], axis=0)

distance_pixels = np.sqrt(np.sum((center1 - center2) ** 2))

# conv dist to cm 
pixel_width = tags.shape[1]  # width of the image in pixels
distance_cm = (distance_pixels / pixel_width) * 3
print(f"{distance_cm:.2f} centimeters.")






cv2.waitKey(0)

cv2.destroyAllWindows()