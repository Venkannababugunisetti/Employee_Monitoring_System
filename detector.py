from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Returns dict with persons, devices, phones
        """
        results = self.model(frame, conf=0.25, verbose=False, classes=[0, 62, 63, 67])
        
        persons = []
        devices = []
        phones = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls == 0:
                    persons.append((x1, y1, x2, y2))
                elif cls in [62, 63]:
                    devices.append((x1, y1, x2, y2))
                elif cls == 67:
                    phones.append((x1, y1, x2, y2))

        return {"persons": persons, "devices": devices, "phones": phones}
