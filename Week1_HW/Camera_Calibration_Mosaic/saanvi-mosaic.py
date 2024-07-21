import cv2
import numpy as np
import glob

def load_calibration(calibration_file):
    # Load calibration data from the file
    with np.load(calibration_file) as data:
#  Extract camera matrix and distortion coefficients
        camera_matrix = data['mtx']
        dist_coeffs = data['dist']
#  Return camera matrix and distortion coefficients
    return camera_matrix, dist_coeffs

def undistort_image(image, camera_matrix, dist_coeffs):
    # Get image dimensions (height, width)
    h, w = image.shape[:2]
    # Compute new camera matrix for undistortion
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    # Undistort the image
    undistorted_img = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    # Crop the undistorted image using ROI
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    return undistorted_img

def harris_corner_detection(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    # Apply Harris corner detection
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # Dilate corners
    dst = cv2.dilate(dst, None)
    # Mark corners on the image
    image[dst > 0.01 * dst.max()] = [0, 0, 255]
    # Return image with marked corners and detected corners
    return image, np.argwhere(dst > 0.01 * dst.max())

def match_features(image1, image2):
    # Detect keypoints and descriptors in image1 using SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    # Detect keypoints and descriptors in image2 using SIFT
    kp2, des2 = sift.detectAndCompute(image2, None)
    # Match descriptors using brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    # Extract matched points from both images
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
    # Return matched points from image1 and image2
    return points1, points2

def create_mosaic(images, camera_matrix, dist_coeffs):
    # Undistort all images using undistort_image function
    undistorted_images = [undistort_image(img, camera_matrix, dist_coeffs) for img in images]
    # Initialize mosaic with the first undistorted image
    mosaic = undistorted_images[0]
    for i in range(1, len(undistorted_images)):
        # Detect Harris corners in both mosaic and current image using harris_corner_detection
        mosaic_corners_img, mosaic_corners = harris_corner_detection(mosaic)
        current_image_corners_img, current_image_corners = harris_corner_detection(undistorted_images[i])
        # Match features between mosaic and current image using match_features
        points1, points2 = match_features(mosaic, undistorted_images[i])
        # Estimate homography using matched points
        homography, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
        # Warp mosaic image using the estimated homography
        height, width, _ = mosaic.shape
        warped_image = cv2.warpPerspective(undistorted_images[i], homography, (width, height))
        # Blend current image into mosaic
        mask_warped_image = np.zeros_like(mosaic, dtype=np.uint8)
        mask_warped_image[warped_image > 0] = warped_image[warped_image > 0]
        mosaic = cv2.addWeighted(mosaic, 1.0, mask_warped_image, 1.0, 0)
    # Return final mosaic image
    return mosaic

# Main:
calibration_file = 'camera_calibration.npz'
camera_matrix, dist_coeffs = load_calibration(calibration_file)

# Load images from specified directory
image_files = glob.glob('/home/saanvi/bwsi-uav2/laboratory_2024/week_1_Hw/camera_calibration_photo_mosaic/International Village - 15 Percent Overlap/*.jpg')
images = [cv2.imread(file) for file in image_files]

# Create mosaic using create_mosaic function
mosaic_image = create_mosaic(images, camera_matrix, dist_coeffs)

# Save the mosaic image to a file
cv2.imwrite('mosaic_image.jpg', mosaic_image)

# Display the mosaic image
cv2.imshow('Mosaic', mosaic_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
