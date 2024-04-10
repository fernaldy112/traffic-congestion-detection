import cv2
import numpy as np

def lk_optical_flow(old_frame, new_frame, old_points, clear_frame):  
  lk_params = dict(winSize = (15,15), 
                   maxLevel = 2, 
                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
  color = np.random.randint(0, 255, (100, 3))
  mask = np.zeros_like(old_frame)
  
  new_points, status, error = cv2.calcOpticalFlowPyrLK(old_frame, new_frame, old_points, None, **lk_params)
  
  good_new = new_points[status == 1]
  good_old = old_points[status == 1]
  
  velocities = good_new - good_old
  speeds = np.linalg.norm(velocities, axis=1)
  average_speed = np.average(speeds)
  
  return average_speed
  
  # TODO: delete
  # frame = clear_frame.copy() 
  # for i, (new, old) in enumerate(zip(good_new, good_old)):
  #   a, b = new.ravel()
  #   c, d = old.ravel()
  #   a, b, c, d = int(a), int(b), int(c), int(d)
  #   mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
  #   frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    
  # mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
  # img = cv2.add(frame, mask_color)
    
  # return img
