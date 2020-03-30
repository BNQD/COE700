import cv2
import numpy as np
import imutils
from queue import Queue
from time import sleep
from statistics import mean
import pandas
import math

fps = 30;

time_saved = 0.5
stable_counter = 0;
#5 seconds
vals = []
reset_time = 0.5 #Number of seconds between instructions -- To be removed after instructions added for robot movement

x_threshold = 10 #Should be calculated based on fps and time_saved
y_threshold = 10 #Should be calculated based on fps and time_saved

def direction_check (x_average, x_threshold, y_average, y_threshold, x, y):
    if (x_average > x_threshold or x < 50):
        return ("right");

    elif (x_average < -x_threshold or x > 580):
        return ("left");

    elif (y_average < -y_threshold or y > 400):
        return ("down")

    elif (y_average > y_threshold or y < 50):
        return ("up");

    else:
        return ("none");

def main():
    center = None
    cap = cv2.VideoCapture(0)
    movement_flag = 0;

    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False

    while ret:
        #Limit to a certain fps
        sleep(1/fps);

        ret, frame = cap.read()
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Hue saturation value

        #Cyan colour to match glove
        low = np.array([75, 120, 120])
        high = np.array([100, 255, 255])

        #Take region of video which matches low/high range of specified colour
        image_mask = cv2.inRange(hsv, low, high)

        #--Not very needed
        #image_mask = cv2.erode(image_mask, None, iterations=6)
        #image_mask = cv2.dilate(image_mask, None, iterations=4)

        output = cv2.bitwise_and(frame, frame, mask=image_mask)

        ###
        #Finding largest contour of the specified colour
        ###
        cnts = cv2.findContours(image_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)



        radius = None
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)

            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            try:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            except:
                pass

            #Min radius of the contour
            if radius > 20:
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
            else:
                #In the case that contour was not found -- Reset
                center = None
                del vals[:]

        if (center is not None):
            if (len(vals) > fps * time_saved): #Limit center arrays to fps * time_saved
                temp_flag = 0;
                direction = direction_check(x_average, x_threshold, y_average, y_threshold, x, y);
                print (direction)

                if (direction != "none"):
                    movement_flag = 1;
                else:
                    del vals[0];

                try:
                    roi = frame[math.floor(x-100):math.floor(x+100), math.floor(y-100):math.floor(y+100)]
                    epsilon = 0.001 * cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, epsilon, True)

                    # make convex hull around hand
                    hull = cv2.convexHull(c)

                    # define area of hull and area of hand
                    areahull = cv2.contourArea(hull)
                    areacnt = cv2.contourArea(c)

                    # find the percentage of area not covered by hand in convex hull
                    arearatio = ((areahull - areacnt) / areacnt) * 100

                    # find the defects in convex hull with respect to hand
                    hull = cv2.convexHull(approx, returnPoints=False)
                    defects = cv2.convexityDefects(approx, hull)

                    # l = no. of defects
                    l = 0

                    # code for finding no. of defects due to fingers
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(approx[s][0])
                        end = tuple(approx[e][0])
                        far = tuple(approx[f][0])
                        pt = (100, 180)

                        # find length of all sides of triangle
                        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                        s = (a + b + c) / 2
                        ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

                        # distance between point and convex hull
                        d = (2 * ar) / a

                        # cosine rule
                        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                        # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                        if angle <= 90 and d > 20:
                            l += 1
                            cv2.circle(roi, far, 3, [255, 0, 0], -1)

                        cv2.line(roi, start, end, [0, 255, 0], 2)

                    l += 1 #Number of fingers


                except Exception as e:
                    pass
                ###################



            try:
                if (movement_flag == 0):
                    vals.append([center[0], center[1], round(radius, 2)])
                    np_lists = np.asarray(vals);
                    x_average = (np.mean(np_lists[:-1, 0] - np_lists[1:, 0]));
                    y_average = (np.mean(np_lists[:-1, 1] - np_lists[1:, 1]))
                else:
                    print ("here");
                    del vals[:];
                    center = None;
                    np_lists = np.asarray(vals);
                    x_average = 0;
                    y_average = 0;
                    sleep(reset_time);
                    movement_flag = 0;

            except:
                pass
        #After averaging, take the one with larger value - Use radius to scale values


        cv2.imshow("Original", frame)
        cv2.imshow("Mask", image_mask)
        cv2.imshow("Output", output)

        if cv2.waitKey(1) == 27:  # exit on ESC
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
