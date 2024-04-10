import numpy as np

def xyxy_to_vertices(xyxys: np.ndarray):
  """
  Returns array of vertices extracted from xyxy format
  
  Args:
    xyxys (ndarray): 2D array representing boxes in xyxy format
    
  Returns:
    ndarray: Array of vertices in shape of (N, 1, 2) representing bounding boxes
  """
  points = np.array([], dtype=np.float32)
  for xyxy in xyxys:
    top_left = [xyxy[0], xyxy[1]]
    top_right = [xyxy[2], xyxy[1]]
    bottom_right = [xyxy[2], xyxy[3]]
    bottom_left = [xyxy[0], xyxy[3]]
    points = np.append(points, [top_left, top_right, bottom_right, bottom_left])
  return np.array(points).reshape(-1, 1, 2)


