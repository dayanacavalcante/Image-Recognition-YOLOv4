# Imports
from os import read
import cv2
import time

# Classes
colors = [(0,255,255), (255,255,0), (0,255,0), (255,0,0)]

# Load the classes
class_names = []
with open("C:\\Users\\RenanSardinha\\Documents\\Data Science\\Projects\\Image-Recognition\\darknet\\coco.names.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Video
cap = cv2.VideoCapture("Video (2).mp4")

# Load Neural Network Weights
# net = cv2.dnn.readNet("weights/yolov4.weights", "cfg/yolov4.cfg")
net = cv2.dnn.readNet("./darknet/yolov4-tiny.weights", "./darknet/yolov4-tiny.cfg")

# Set Neural Network Parameters
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(608,608), scale=1/255)

# Read all video frames
while True:
    
    # frame capture
    _, frame = cap.read()

    # Start of MS count
    start = time.time()

    # Detection
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    # End of MS count
    end = time.time()

    # Run through all detections
    for (classid,score,box) in zip(classes, scores, boxes):

        # Generate a color for the class
        color = colors[int(classid) % len(colors)]

        # Get the class name by id and its accuracy score
        label = f"{class_names[classid[0]]} : {score}"

        # Draw the detection box
        cv2.rectangle(frame,box,color,2)

        # Write the class name on top of the object's box
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculate the time it took to detect
    fps_label = f"FPS: {round((1.0/(end - start)),2)}"

    # Write the FPS on the image
    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)
    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    # Show the image
    cv2.imshow("detections", frame)

    # Wait from answer 
    if cv2.waitKey(1)==27:
        break