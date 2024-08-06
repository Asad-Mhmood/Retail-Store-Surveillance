from ultralytics import YOLO
import math
import cv2
import mysql.connector
#con=mysql.connector.connect(host="localhost",user="root",password="",database="analytics")
#cursor = con.cursor()

# Load a model
model = YOLO('best.pt')  # pretrained YOLOv8n model
cap = cv2.VideoCapture('videos\gunman-in-store.mp4')  # Add Video Path



classNames = [
    'Gun'

]


while True:

    success,img = cap.read()
    img = cv2.resize(img,(1600,900))
    #results = model(img,stream=True)
    results = model.track(img, tracker="bytetrack.yaml")  # with ByteTrack

    # Process results list

    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs

        for box in boxes:
            print(box)

            # Bounding Box

            #print("Box",box)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            #print(x1, y1, x2, y2)




            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]



            # Confidence
            conf = math.ceil((box.conf[0] * 100))



            # ID
            if box.id is not None:
                id = int(box.id[0])


            if conf>20:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(img, currentClass, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), thickness=2)
                cv2.putText(img, str(conf), (x1 + 90, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255),thickness=2)
                cv2.putText(img, str(id), (x1 + 180, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255),thickness=2)


                ## WhatsApp Alert
                '''# pip install pywhatkit
                import pywhatkit as kit
                # Specify the phone number (with country code) and the message
                phone_number = "+923214110555"
                message = "Gun detected"
                # Send the message instantly
                kit.sendwhatmsg_instantly(phone_number, message)'''




        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cap.destroyAllWindows()
