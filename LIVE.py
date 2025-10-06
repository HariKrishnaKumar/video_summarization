# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["yolo-coco/coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["yolo-coco/yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco/yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i- 1] for i in net.getUnconnectedOutLayers()]


def LiveRecognition(query):

    for img in os.listdir('static/unique'):
        os.remove('static/unique/'+img)
        
    for img in os.listdir('static/common'):
        os.remove('static/common/'+img)
        
    # initialize the video stream
    vs = cv2.VideoCapture(0)
    (W, H) = (None, None)

    #Read width and height of video frame
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    threshold = 20.

    #Create video result
    writer = cv2.VideoWriter(r'D:\2023_PROJECTS\Final_video_summarization\Final_video_summarization\video_summarization\static\outputvideo\1.mp4',cv2.VideoWriter_fourcc(*'H264'), 25, (width, height))
    ret, frame1 = vs.read()
    prev_frame = frame1

    a = 0 #Total frames are trained
    b = 0 #Unique frames are kept
    c = 0 #Common frame
    prev_frame = frame1
    # loop over frames from the video stream
    while True:
        print(query)
        # read the next frame from the video stream
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # initialize list to store objects detected in current frame
        objects = []

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs
        boxes = []
        confidences = []
        classIDs = []
        print(W,H)

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

                    # update list of detected objects
                    objects.append((x, y, x + int(width), y + int(height)))

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
            args["threshold"])

        # ensure at least one detection exists
        if len(idxs) > 0:
            print('dfghjkljgchfcghjkj')
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # print(i)
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                
                
                text1="{}".format(LABELS[classIDs[i]])
                print(text1)
                if text1 in query:
                    print('hghghghjg')
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    #Check if it a Unique frame? Save it
                    if (((np.sum(np.absolute(frame - prev_frame)) / np.size(frame)) > threshold)):
                        writer.write(frame)
                        prev_frame = frame
                        a += 1
                        cv2.imwrite('static/unique/unique'+str(a)+'.jpg', frame)
                    else:
                        prev_frame = frame
                        b += 1
                        cv2.imwrite('static/common/common'+str(b)+'.jpg', frame)


    ##    # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    
    
    #Print result
    print("Total frames: ", c)
    print("Unique frames: ", a)
    print("Common frames: ", b)
    #save result

    writer.release()
    # release the file pointers
    print("[INFO] cleaning up...")
    vs.release()

    cv2.destroyAllWindows()
