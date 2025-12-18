import cv2

def draw_status(frame, status):
    """Draws a general status string on the frame."""
    cv2.putText(frame, f"Status: {status}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


def iou(box1, box2):
    """Calculates Intersection over Union of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0
