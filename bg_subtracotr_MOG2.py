import cv2
from tracker.sort import Sort
import numpy as np
from time import time

class GetFPS:
    def __init__(self) -> None:
        self.prev_time = 0
        self.curr_time = 0

    def get(self):
        self.curr_time = time()
        fps = 1/(self.curr_time-self.prev_time)
        self.prev_time = self.curr_time
        return int(fps)

    def draw_in_img(self, img, scale=1):
        cv2.rectangle(img, (0, 0), (int(200*scale), int(50*scale)), (100, 46, 21), cv2.FILLED)
        cv2.putText(
            img, f'FPS: {self.get()}', (int(10*scale), int(40*scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5*scale, (255, 255, 255), int(2*scale)
        )
        return img


cap = cv2.VideoCapture('./assets/highway.mp4')
detect_change = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=10)
roi_mask = cv2.imread('./assets/highway.mp4.mask.png')

tracker = Sort(max_age=10, min_hits=5)
fps = GetFPS()

while True:
    read, frame = cap.read()

    key = cv2.waitKey(1)
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    elif key == ord(' '): # pause
        cv2.waitKey(0)
    
    if not read:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
        print('continue')
        continue

    frame = cv2.bitwise_and(frame, roi_mask)

    mask = detect_change.apply(frame)
    # 0 if pixel <=254 else 255
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    countors, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    detections = np.empty((0, 5))

    for cnt in countors:
        area = cv2.contourArea(cnt)
        if area > 150 and area < 5000:
            x1, y1, w, h = cv2.boundingRect(cnt)
            detections = np.vstack([
                detections,
                np.array([x1, y1, x1+w, y1+h, 0.86])
            ])
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # if key == ord('r'):
    res_tracker = tracker.update(detections)

    for res in res_tracker:
        x1, y1, x2, y2, id = res.astype(np.int32)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(id), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    fps.draw_in_img(frame)
    cv2.imshow('frame', frame)
