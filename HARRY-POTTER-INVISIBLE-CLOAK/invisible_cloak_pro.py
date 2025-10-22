import cv2
import numpy as np
import time

def nothing(x):
    pass

# Create a window for trackbars (debugging HSV mask)
cv2.namedWindow("Mask Controls")
cv2.resizeWindow("Mask Controls", 400, 300)

cv2.createTrackbar("Lower-H", "Mask Controls", 0, 180, nothing)
cv2.createTrackbar("Lower-S", "Mask Controls", 120, 255, nothing)
cv2.createTrackbar("Lower-V", "Mask Controls", 70, 255, nothing)
cv2.createTrackbar("Upper-H", "Mask Controls", 10, 180, nothing)
cv2.createTrackbar("Upper-S", "Mask Controls", 255, 255, nothing)
cv2.createTrackbar("Upper-V", "Mask Controls", 255, 255, nothing)

# Second red range
cv2.createTrackbar("Lower-H2", "Mask Controls", 170, 180, nothing)
cv2.createTrackbar("Upper-H2", "Mask Controls", 180, 180, nothing)

cap = cv2.VideoCapture(0)

print("Hold still for 3 seconds - capturing background...")
time.sleep(3)
ret, background = cap.read()
background = np.flip(background, axis=1)

prev_frame_time = 0

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = np.flip(img, axis=1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Trackbar values (adjusts red detection live)
    l_h = cv2.getTrackbarPos("Lower-H", "Mask Controls")
    l_s = cv2.getTrackbarPos("Lower-S", "Mask Controls")
    l_v = cv2.getTrackbarPos("Lower-V", "Mask Controls")
    u_h = cv2.getTrackbarPos("Upper-H", "Mask Controls")
    u_s = cv2.getTrackbarPos("Upper-S", "Mask Controls")
    u_v = cv2.getTrackbarPos("Upper-V", "Mask Controls")
    l_h2 = cv2.getTrackbarPos("Lower-H2", "Mask Controls")
    u_h2 = cv2.getTrackbarPos("Upper-H2", "Mask Controls")

    lower_red1 = np.array([l_h, l_s, l_v])
    upper_red1 = np.array([u_h, u_s, u_v])
    lower_red2 = np.array([l_h2, l_s, l_v])
    upper_red2 = np.array([u_h2, u_s, u_v])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    inverse_mask = cv2.bitwise_not(mask)
    cloak_area = cv2.bitwise_and(background, background, mask=mask)
    current_area = cv2.bitwise_and(img, img, mask=inverse_mask)
    final_output = cv2.addWeighted(cloak_area, 1, current_area, 1, 0)

    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time + 1e-8))
    prev_frame_time = new_frame_time
    cv2.putText(final_output, f"FPS: {fps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Invisible Cloak Output", final_output)
    cv2.imshow("Mask View (Debug)", mask)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
