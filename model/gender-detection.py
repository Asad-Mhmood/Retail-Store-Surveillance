from ultralytics import YOLO
import math
import cv2
import mysql.connector
con=mysql.connector.connect(host="localhost",user="root",password="",database="analytics")
cursor = con.cursor()

# Load a model
model = YOLO('gender.pt')  # pretrained YOLOv8n model
cap = cv2.VideoCapture('videos\gender1.mp4') #Add Video Path



classNames = [
    'Female','Male'

]
limits = [140,100,140,850]
counter={
    'Male':0,
    'Female':0,
}

totalcounts=[]


while True:

    success,img = cap.read()
    #img = cv2.resize(img,(1600,900))
    #results = model(img,stream=True)
    results = model.track(img, tracker="bytetrack.yaml")  # with ByteTrack

    # Process results list

    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
        for box in boxes:
            print(box)

            # Bounding Box

            #print("Box",box)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            #print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)


            # Confidence
            conf = math.ceil((box.conf[0] * 100))
            # cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(20,y1)),scale=1,thickness=1)
            #cv2.putText(img,conf,(x1, y1),2)
            #print(conf)
            cv2.putText(img, str(conf), (x1+90, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), thickness=2)


            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            cv2.putText(img, currentClass, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), thickness=2)


            # ID
            if box.id is not None:
                id = int(box.id[0])
                cv2.putText(img, str(id), (x1+180, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), thickness=2)

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)


            if limits[1] < cy < limits[3] and limits[2] - 6 < cx < limits[2] + 6:
                # if currentClass == 'car' or currentClass == 'motorcycle' or currentClass == 'bus' or currentClass == 'truck' or currentClass=='bicycle':

                #if totalcounts.count(id) == 0:
                    #totalcounts.append(id)
                counter[currentClass] += 1
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (255, 255, 255), 5)
                cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

                if currentClass == 'Male':
                    cursor.execute("""INSERT INTO `footfall` (`Male`,`Female`)VALUES
                    ('{}','{}')""".format(1, 0))
                    con.commit()

                if currentClass == 'Female':
                    cursor.execute("""INSERT INTO `footfall` (`Male`,`Female`)VALUES
                    ('{}','{}')""".format(0, 1))
                    con.commit()

        y_offset = 80
        for vehicle_type, count in counter.items():
            cv2.putText(img, f'{vehicle_type.capitalize()} : {count}', (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), thickness=2)
            y_offset += 50

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cap.destroyAllWindows()
