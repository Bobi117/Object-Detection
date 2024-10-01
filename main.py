import cv2
import numpy as np


# Create tracker object
class EuclideanDistTracker:
    def __init__(self):
        self.objects = []
        self.object_id = 0

    def update(self, detections):
        new_objects = []
        ids = []

        for detection in detections:
            x, y, w, h = detection
            assigned = False

            for obj in self.objects:
                ox, oy, ow, oh, obj_id = obj
                if self._euclidean_distance(x, y, ox, oy) < 50:  # Arbitrary threshold
                    new_objects.append([x, y, w, h, obj_id])
                    ids.append(obj_id)
                    assigned = True
                    break

            if not assigned:
                new_objects.append([x, y, w, h, self.object_id])
                ids.append(self.object_id)
                self.object_id += 1

        self.objects = new_objects
        return [(x, y, w, h, obj_id) for (x, y, w, h, obj_id) in new_objects]

    def _euclidean_distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
# Main code
cap = cv2.VideoCapture("highway.mp4")
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
tracker = EuclideanDistTracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    roi = frame[340: 720, 500: 800]

    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
