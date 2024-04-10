import supervision as sv

def count_object(detections: sv.Detections):
  """
  Returns number of bounding boxes
  
  Args:
    detections (Detections): Result of object detection with YOLOv8
    
  Returns:
    int: Length of the detections
  """
  return len(detections)