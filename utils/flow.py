import supervision as sv

def count_object(detections: sv.Detections) -> int:
  """
  Returns number of bounding boxes.
  
  Parameters
  ----------
  detections : Detections 
    Result of object detection with YOLOv8.
    
  Returns
  -------
  int
    Length of the detections.
  """
  return len(detections)