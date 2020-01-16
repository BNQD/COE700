#https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import numpy as np


#MReturns image masked with hsv range
def mask_with_colour(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Hue saturation value
    low = np.array([0, 60, 0])  # 0 45 0
    high = np.array([45, 255, 150])  # 25 255 180
    frame_mask = cv2.inRange(hsv, low, high)
    #cv2.imshow("Frame_2", frame)
    frame = cv2.bitwise_and(frame, frame, mask=frame_mask)
    #cv2.imshow("Color Masked", frame)
    return frame;

def count_non_black_pixels (frame):
    n_non_black_pixels = np.sum(frame != 0);
    return n_non_black_pixels;



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="csrt",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]

# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(args["tracker"].upper())

# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# initialize the FPS throughput estimator
fps = None
# initialize other vars
prev_x = 0
prev_y = 0
start_y = y = 100
start_x = x = 150
start_h = start_w = 150
horz_counter = 0
vert_counter = 0
dir_x = "none"
dir_y = "none"
coloured_frame_counter = 0;
prev_loc = "";


# loop over frames from the video stream
while True:
    h = w = 150
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = mask_with_colour(frame)

    frame = frame[1] if args.get("video", False) else frame

    # check to see if we have reached the end of the stream
    if frame is None:
        break

    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]

    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)

        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)

        # update the FPS counter
        fps.update()
        fps.stop()

        counter = 0;

        current_loc = "starting"
        if (x < 0):
            current_loc = "left"
        elif (x > 300):
            current_loc = "right"
        elif (y > 250):
            current_loc = "down"
        elif (y < 0):
            current_loc = "up"
        else:
            current_loc = "starting"

        if (current_loc != "starting"):
            prev_loc = current_loc;
            initBB = None;
        prev_x = x
        prev_y = y

        # initialize the set of information we'll be displaying on
        # the frame


        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
            ("Top Left", "{:d}, {:d}".format(x, y)),
            ("Location", current_loc),
            ("Prev Loc", prev_loc)
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        y = y if y > 0 else 0
        x = x if x > 0 else 0
        crop_img = frame[y:y+h, x:x+w]
        cv2.imshow("cropped", crop_img)

        if (not success or key == ord('r')):
            tracker.init(frame, initBB)
            fps = FPS().start()
    else: #check cropped image
        frame_cropped = frame[start_y:start_y+start_h, start_x:start_x+start_w]
        frame_masked = mask_with_colour(frame_cropped)
        cv2.imshow("Masked Colour Counter", frame_masked)
        count_non_black = count_non_black_pixels(frame_masked)
        if (count_non_black > 5000):
            coloured_frame_counter+=1;
        else:
            coloured_frame_counter = 0;
    print(coloured_frame_counter)

    # show the output frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if (coloured_frame_counter > 500):
        coloured_frame_counter = 0;
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = (start_x, start_y, start_x+round(start_w/4), start_y+round(start_h/4))

        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()
        # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()

# otherwise, release the file pointer
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()