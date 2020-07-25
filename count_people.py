from persondetection import TrackableObject,CentroidTracker,DetectorAPI
import cv2
import numpy as np


if __name__ == "__main__":
    
    path="I:/person-detection-tf-api-flask-master/test.Mp4"
    odapi = DetectorAPI()
    threshold = 0.7
    cap = cv2.VideoCapture(path)
    
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackableObjects = {}
    
    h = 440
    w = 590


    line_up = int(2*(h/2.2))
    line_down =int(3*(h/3.5))    
    up_limit = int(1*(h/1.1))
    down_limit = int(4*(h/5))
    
    line_down_color = (255,0,0)
    line_up_color = (0,0,255)
    pt1 =  [0, line_down];
    pt2 =  [w, line_down];
    pts_L1 = np.array([pt1,pt2], np.int32)
    pts_L1 = pts_L1.reshape((-1,1,2))
    pt3 =  [0, line_up];
    pt4 =  [w, line_up];
    pts_L2 = np.array([pt3,pt4], np.int32)
    pts_L2 = pts_L2.reshape((-1,1,2))
    
    pt5 =  [0, up_limit];
    pt6 =  [w, up_limit];
    pts_L3 = np.array([pt5,pt6], np.int32)
    pts_L3 = pts_L3.reshape((-1,1,2))
    pt7 =  [0, down_limit];
    pt8 =  [w, down_limit];
    pts_L4 = np.array([pt7,pt8], np.int32)
    pts_L4 = pts_L4.reshape((-1,1,2))

    font = cv2.FONT_HERSHEY_SIMPLEX
  
  
    totalDown=0
    totalUp =0
        
  
    while True:
        r, img1 = cap.read()
        img = cv2.resize(img1, (800, 600))
        
        frameHeight = img.shape[0]
        frameWidth = img.shape[1]
        boxes, scores, classes, num = odapi.processFrame(img)

        pts = np.array([[180,150],  
                    [410, 160],
                    [800, 550]], 
                   np.int32)       
        rect= []
        
        for i in range(len(boxes)):

            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                rect.append((box[1],box[0],box[3],box[2]))
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                
        objects  =ct.update(rect)        
       
        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)
     
            if to is None:
                to = TrackableObject(objectID, centroid)
     
            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)
     
                if not to.counted:
   
                    if direction < 0 and centroid[1] in range(line_up, line_down):
                        totalUp += 1
                        to.counted = True
     
                    elif direction > 0 and centroid[1] in range(line_down, line_up):
                        totalDown += 1
                        to.counted = True
                
     
            trackableObjects[objectID] = to

            cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        info = [
            ("Up", totalUp),
            ("Down", totalDown),
            ]
        print("Up", totalUp)
        print("Down", totalDown)

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(img, text, (10, frameHeight - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        
        img = cv2.polylines(img,[pts_L1],False,line_down_color,thickness=2)
        img = cv2.polylines(img,[pts_L2],False,line_up_color,thickness=2)
        
        img = cv2.polylines(img, [pts],False,  (255, 240, 0), thickness=2)
              

        cv2.imshow("preview", img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break




