import numpy as np
import cv2
import time

# Parameters for Lucas-Kanade optical flow
lk_params = {'winSize':(15, 15), 'maxLevel':2, 'criteria':(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}

# Parameters for corner detection using Shi-Tomasi method
feature_params = {'maxCorners':100, 'qualityLevel':0.3, 'minDistance':7, 'blockSize':7}

# Variables for tracking
trajectory_len = 40      # Length of the trajectory
detect_interval = 2      # Interval for detecting new features
trajectories = []        # List to store trajectories
frame_idx = 0            # Frame index

# Capture video from the first camera device
cap = cv2.VideoCapture(0)

while True:

    # Store the start time to calculate FPS
    start = time.time()

    # Read a new frame from the video capture
    suc, frame = cap.read()
    if not suc:
        break  # Break if there is an issue reading the frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) #Convert frame to grayscale
    img = frame.copy()  # Create a copy of the frame for displaying results

    # Calculate optical flow for the existing trajectories
    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
 
        #Get the last points of trajectories
        lastpoints = np.array([trajectory[-1] for trajectory in trajectories], dtype=np.float32).reshape(-1,1,2)

        # Calculate forward optical flow
        fflow = cv2.calcOpticalFlowPyrLK(img0, img1, lastpoints, None, **lk_params) 
        nextpoints = fflow[0]

        # Calculate backward optical flow
        bflow = cv2.calcOpticalFlowPyrLK(img1, img0, nextpoints, None, **lk_params) 
        backpoints = bflow[0]

        # Calculate difference between forward-backward flow
        diff = abs(lastpoints - backpoints)
        dist = np.linalg.norm(diff, axis=2)

        # Identify good points with small differences
        good_points = dist <= 1.0
        good_points = good_points.reshape(-1)

        
        # Update the trajectories with the new points and draw the new points on the image
        new_trajectories = []
        for i, trajectory in enumerate(trajectories):
            if good_points[i]:
                trajectory.append(nextpoints[i][0])
                new_trajectories.append(trajectory)

        # Make sure to trim down the trajectories when they get too long
        trajectories = [traj[-trajectory_len:] for traj in new_trajectories]

        trajectories = new_trajectories

        # Draw the trajectories on the image
        cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
        cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)

    # Detect new features at specified intervals
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)  # Create an empty mask
        mask[:] = 255  # Set all pixels to 255 (white)

        # Mask out regions where features are already tracked
        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)  # Draw circles on the mask

        # Detect good features to track
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            # Add new features to the trajectories
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    frame_idx += 1  # Increment frame index
    prev_gray = frame_gray  # Update previous frame

    # End time to calculate FPS
    end = time.time()
    # Calculate FPS
    fps = 1 / (end - start)
    
    # Display FPS on the image
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Show the image with optical flow
    cv2.imshow('Optical Flow', img)
    # Show the mask
    cv2.imshow('Mask', mask)

    # Exit on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()