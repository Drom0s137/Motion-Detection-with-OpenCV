# import the necessary packages
from centroidtracker import CentroidTracker
import imutils
import cv2
from collections import OrderedDict


# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# initialize face cascade (Trial for face detection)
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# vs = VideoStream(src=0).start()
#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('vtest.avi')
#cap = cv2.VideoCapture('head-pose-face-detection-female.mp4')
cap = cv2.VideoCapture('worker-zone-detection.mp4')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=True)


# setup track bar
def nothing(x):
    pass


cv2.namedWindow('Adjustable Settings')
cv2.createTrackbar('Area Upper', 'Adjustable Settings', 10000, 10000, nothing)
cv2.createTrackbar('Area Limit', 'Adjustable Settings', 3000, 5000, nothing)
cv2.createTrackbar('Dilate Iterations', 'Adjustable Settings', 3, 5, nothing)

alive = OrderedDict()
usedObj = []

# loop over the frames from the video stream
while True:

    # read the next frame from the video stream and resize it
    ret, frame = cap.read()
    if frame is None:
        break
    frame = imutils.resize(frame, width=400)

    # if the frame dimensions are None, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # track bar input data
    AreaUpper = cv2.getTrackbarPos('Area Upper', 'Adjustable Settings')
    AreaLower = cv2.getTrackbarPos('Area Limit', 'Adjustable Settings')
    iterate = cv2.getTrackbarPos('Dilate Iterations', 'Adjustable Settings')

    # construct a blob from the frame, pass it through the network,
    # obtain our output predictions, and initialize the list of
    # bounding box rectangles
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    blur = cv2.GaussianBlur(fgmask, (5, 5), 0)
    dilated = cv2.dilate(blur, None, iterations=iterate)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rects = []

    # loop over the detections
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        # input validation
        if cv2.contourArea(contour) < AreaLower:
            continue
        elif cv2.contourArea(contour) > AreaUpper:
            continue
        # compute the (x, y)-coordinates of the bounding box for
        # the object, then update the bounding box rectangles list
        box = (x, y, x + w, y + h)
        rects.append(box)
        # draw a bounding box surrounding the object so we can
        # visualize it
        (startX, startY, endX, endY) = box
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        cv2.drawContours(frame, contour, -1, (0, 0, 255), 2)

    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects, disappear = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        if len(alive) == 0 or objectID not in usedObj:
            alive[objectID] = 0
            usedObj.append(objectID)
        elif objectID in usedObj:
            alive[objectID] += 1

        text = "ID {}".format(objectID)
        textDis = "Exist: {}".format(alive[objectID])
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_ITALIC, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, textDis, (30, 90),
                    cv2.FONT_ITALIC, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    for (ObjectID, Dis) in disappear.items():
        text = "ID {}".format(ObjectID)
        cv2.putText(frame, text, (30, 60),
                    cv2.FONT_ITALIC, 0.5, (0, 255, 0), 2)
        textDis = "Disappear: {}".format(Dis)
        cv2.putText(frame, textDis, (30, 30),
                    cv2.FONT_ITALIC, 0.5, (0, 255, 0), 2)


    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
