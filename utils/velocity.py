import cv2
import numpy as np

def lk_optical_flow(old_frame: np.ndarray, new_frame: np.ndarray, old_points: np.ndarray) -> tuple:
  """
  Calculates average speed of points by calculating optical flow for sparse feature set using the iterative Lucas-Kanade method with pyramids.
  
  Parameters
  ----------
  old_frame : ndarray 
    Grayscale image, the frame before new_frame, to be used for optical flow calculation.
  new_frame: ndarray
    Grayscale image, the frame after old_frame, to be used for optical flow calculation.
  old_points: ndarray
    Array in shape of (N, 1, 2) that contains points in the old_frame to be used for calculation.
    
  Returns
  -------
  good_new : ndarray
    New position of each corresponding point in good_old.
  good_old : ndarray
    Points in the old_frame those have their optical flow successfully calculated.
  average_speed : float32
    Average speed of the points calculated from the obtained optical flow.
  """
  lk_params = dict(winSize = (50, 50), 
                   maxLevel = 2, 
                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
  
  new_points, status, error = cv2.calcOpticalFlowPyrLK(old_frame, new_frame, old_points, None, **lk_params)
  
  good_new = new_points[status == 1]
  good_old = old_points[status == 1]
  
  print(type(good_new))
  print(type(good_old))
  
  velocities = good_new - good_old
  speeds = np.linalg.norm(velocities, axis=1)
  average_speed = np.average(speeds)
  
  print(type(average_speed))
  
  return good_new, good_old, average_speed

def visualize_lk(clear_frame: np.ndarray, good_new: np.ndarray, good_old: np.ndarray) -> np.ndarray:
  """
  Returns an image that visualize the movement of points used as input for LK optical flow calculation.
  
  Parameters
  ----------
  clear_frame : ndarray 
    Coloured image that its grayed out image is used as input for LK optical flow calculation.
  good_new : ndarray
    New position of each corresponding point in good_old.
  good_old : ndarray
    Points in the clear_frame those have their optical flow successfully calculated.
    
  Returns
  -------
  ndarray
    Image that visualize the movement of points used as input for LK optical flow calculation.
  """
  color = np.random.randint(0, 255, (100, 3))
  mask = np.zeros_like(clear_frame)

  frame = mask.copy() 
  
  for i, (new, old) in enumerate(zip(good_new, good_old)):
    new_x, new_y = new.ravel()
    old_x, old_y = old.ravel()
    new_x, new_y, old_x, old_y = int(new_x), int(new_y), int(old_x), int(old_y)
    mask = cv2.line(mask, (new_x, new_y), (old_x, old_y), color[i].tolist(), 2)
    frame = cv2.circle(frame, (new_x, new_y), 5, color[i].tolist(), -1)
    
  img = cv2.add(frame, mask)
    
  return img
