import cv2
import numpy as np

def lk_optical_flow(old_frame, new_frame, old_points):  
  lk_params = dict(winSize = (50, 50), 
                   maxLevel = 2, 
                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
  
  new_points, status, error = cv2.calcOpticalFlowPyrLK(old_frame, new_frame, old_points, None, **lk_params)
  
  good_new = new_points[status == 1]
  good_old = old_points[status == 1]
  
  velocities = good_new - good_old
  speeds = np.linalg.norm(velocities, axis=1)
  average_speed = np.average(speeds)
  
  return good_new, good_old, average_speed

def visualize_lk(clear_frame, good_new, good_old):
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
