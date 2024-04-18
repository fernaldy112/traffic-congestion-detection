import numpy as np

def xyxy_to_vertices(xyxys: np.ndarray, vertices_type: str = "midpoint"):
  """
  Returns array of vertices extracted from xyxy format
  
  Args:
    xyxys (ndarray): 2D array in shape of (N, 4) representing boxes in xyxy format
    vertices_type (str): {"midpoint", "corner"}
    
  Returns:
    ndarray: Array of vertices in shape of (N, 1, 2) representing bounding boxes
  """
  if vertices_type == "corner":
    return xyxy_to_corners(xyxys)
  elif vertices_type == "midpoint":
    return xyxy_to_midpoints(xyxys)

def xyxy_to_corners(xyxys: np.ndarray):
  """
  Returns array of corners extracted from xyxy format
  
  Args:
    xyxys (ndarray): 2D array in shape of (N, 4) representing boxes in xyxy format
    
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

def xyxy_to_midpoints(xyxys: np.ndarray):
  """
  Returns array of midpoints extracted from xyxy format
  
  Args:
    xyxys (ndarray): 2D array in shape of (N, 4) representing boxes in xyxy format
    
  Returns:
    ndarray: Array of midpoints in shape of (N, 1, 2) representing bounding boxes
  """
  points = np.array([], dtype=np.float32)
  for xyxy in xyxys:
    midpoint_x = (xyxy[0] + xyxy[2]) / 2
    midpoint_y = (xyxy[1] + xyxy[3]) / 2
    midpoint = np.array([midpoint_x, midpoint_y], dtype=np.float32)
    points = np.append(points, [midpoint])
  return np.array(points).reshape(-1, 1, 2)


