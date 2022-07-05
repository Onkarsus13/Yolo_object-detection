import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3-tiny.cfg', 'yolov3-tiny.weights')
classes = []

with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

img = cv2.imread('person.jpg')
#cap = cv2.VideoCapture(0)

#img = cv2.imread('dog.jpg')

print(img.shape)
hight, width, _ = img.shape
blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), swapRB=True, crop=True)
net.setInput(blob)

l = net.getUnconnectedOutLayers()


layeroutputs = net.forward(l)
boxes = []
confidences = []
class_ids = []

for output in layeroutputs:
    for detection in output:

        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            centre_x = int(detection[0] * width)
            centre_y = int(detection[1] * hight)
            w = int(detection[2] * width)
            h = int(detection[3] * hight)

            x = int(centre_x - w / 2)
            y = int(centre_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

#In case if you want to print out how many objects have been detected,
number_objects_detected = len(boxes)
print("Total number of objects detected is:", len(boxes))

indexes = cv2.dnn.NMSBoxes(boxes , confidences ,0.5 ,0.4)
font  = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0,255 , size =(len(boxes) , 3))

#print(indexes)

for i in indexes.flatten():
    x,y,w,h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = str(round(confidences[i],2))
    color = colors[i]
    cv2.rectangle(img , (x,y) , (x+w , y+h) , color ,2 )
    #cv2.putText(img ,label + "" +confidence  , (x,y+20),font , 2,(255 ,255,255) ,2)
    #You can have different color and label for different objects detected using below line
    cv2.putText(img ,label + "" + confidence  , (x,y+20), font , 2, color ,2)

cv2.imshow('Image' , img)
cv2.waitKey(0)
cv2.destroyAllWindows()

