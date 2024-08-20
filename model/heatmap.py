from ultralytics import YOLO
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model
cap = cv2.VideoCapture('model/videos/gender1.mp4')

classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Map layout coordinates
map_pts = np.float32([[5, 7], [1556, 7], [6, 780], [1556, 780]])
# Camera view coordinates
camera_pts = np.float32([[4, 6], [1581, 7], [6, 857], [1582, 859]])

# Compute the perspective transform matrix
matrix = cv2.getPerspectiveTransform(camera_pts, map_pts)

def map_to_layout(x, y, matrix):
    points = np.array([[x, y]], dtype='float32')
    points = np.array([points])
    transformed_points = cv2.perspectiveTransform(points, matrix)
    return int(transformed_points[0][0][0]), int(transformed_points[0][0][1])

# Load the store layout image
layout_img = cv2.imread('layout1.PNG')
layout_height, layout_width, _ = layout_img.shape

# Initialize heatmap overlay
heatmap = np.zeros((layout_height, layout_width), dtype=np.float32)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (1600, 900))
    results = model.track(img, tracker="bytetrack.yaml")  # with ByteTrack

    # Decay the heatmap over time
    heatmap *= 0.95

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            conf = math.ceil((box.conf[0] * 100))
            cv2.putText(img, str(conf), (x1+90, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), thickness=2)

            cls = int(box.cls[0])
            currentClass = classNames[cls]
            cv2.putText(img, currentClass, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), thickness=2)

            if box.id is not None:
                id = int(box.id[0])
                cv2.putText(img, str(id), (x1+180, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), thickness=2)

            # Calculate the center point of the bounding box
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Map the center point to the map layout
            map_x, map_y = map_to_layout(center_x, center_y, matrix)
            print(f'Person ID {id} at map coordinates: ({map_x}, {map_y})')

            # Update heatmap with a Gaussian blob instead of a single point
            cv2.circle(heatmap, (map_x, map_y), 15, (1), -1)

    # Apply Gaussian blur to the heatmap
    heatmap_blurred = cv2.GaussianBlur(heatmap, (31, 31), 0)

    # Normalize the heatmap
    heatmap_normalized = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8
    heatmap_uint8 = np.uint8(heatmap_normalized)

    # Apply a colormap to the heatmap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Overlay the heatmap on the layout image
    overlay = cv2.addWeighted(layout_img, 0.6, heatmap_colored, 0.4, 0)

    # Display the overlay
    cv2.imshow("Layout with Heatmap", overlay)

    # Display the original image with detections
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Display the heatmap with a color bar
def display_heatmap_with_colorbar(heatmap):
    plt.imshow(heatmap, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.show()

# Example usage
heatmap_for_display = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX)
display_heatmap_with_colorbar(heatmap_for_display)
