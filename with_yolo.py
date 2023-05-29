import cv2
from ultralytics import YOLO
import numpy as np
from tracker.sort import Sort

cap = cv2.VideoCapture('./assets/cars.mp4')

model = YOLO('./assets/yolov8m.pt')

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

mask = cv2.imread('./assets/cars.mp4.mask.png')
tracker = Sort()
cross_pnt = np.array([400, 297, 673, 297])
id_counter = []
curr_count = 0  # to keep count for the last count

while True:
    _, img = cap.read()
    img_roi = cv2.bitwise_and(img, mask)
    results = model(img_roi, stream=True)
    detections = np.empty((0, 5))

    for res in results:
        for box in res.boxes:
            cls = int(box.cls[0].numpy())
            class_name = classNames[cls]

            if class_name in ['car', 'truck', 'bus', 'motorbike']:
                x1, y1, x2, y2 = box.xyxy[0].numpy().astype(np.int32)
                confidence = box.conf[0].numpy()
                cv2.putText(
                    img, class_name, (x1, y1-10),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1
                )

                detections = np.vstack([
                    detections,
                    np.array([x1, y1, x2, y2, confidence])
                ])

    res_tracker = tracker.update(detections)

    cv2.line(img, cross_pnt[:2], cross_pnt[2:], (0, 0, 255), 2)

    for res in res_tracker:
        x1, y1, x2, y2, id = res.astype(np.int32)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if (y2 >= cross_pnt[1]) and (id not in id_counter):
            cv2.line(img, cross_pnt[:2], cross_pnt[2:], (0, 255, 0), 2)
            id_counter.append(id)
            curr_count += 1

    cv2.putText(
        img, f'Count: {curr_count}', (190, 85),
        cv2.FONT_HERSHEY_COMPLEX, 1.1, (0, 0, 0), 2
    )

    cv2.imshow('img', img)
    if cv2.waitKey(1000//30) == ord('q'):
    # if cv2.waitKey(0) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
