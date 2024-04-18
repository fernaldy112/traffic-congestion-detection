import cv2
import numpy as np
import ultralytics

def manual_mask(frame: np.ndarray, vertices: np.ndarray) -> np.ndarray:
  """
  Masks a frame using manually defined vertices.
  
  Parameters
  ----------
  frame : ndarray 
    The frame to be masked.
  vertices : ndarray
    A list of vertex in shape of (N, 2) that define a polygon to mask the frame.
    
  Returns
  -------
  ndarray
    Masked frame.
  """
  mask = np.zeros(frame.shape[:2], dtype=np.uint8)
  cv2.fillPoly(mask, [vertices], 255)
  non_zero_pixels = cv2.countNonZero(mask)
  result = cv2.bitwise_and(frame, frame, mask=mask)
  return result, non_zero_pixels

def get_yolov8_mask(frame: np.ndarray, model: ultralytics.YOLO) -> np.ndarray:
  """
  Returns a mask for a frame predicted by yolov8 model.
  
  Parameters
  ----------
  frame : ndarray 
    The frame used as input for the prediction model.
  model : YOLO
    A yolov8 segmentation model used to predict mask for the input frame.
    
  Returns
  -------
  ndarray
    Mask for the input frame.
  """
  result = model(frame)[0]
  height, width, _ = frame.shape
  masks = [cv2.resize(mask.numpy()*255, (width, height)).astype(np.uint8) for mask in result.masks.data]
  merged_mask = merge_masks(masks, (height, width))
  return merged_mask

def yolov8_mask(frame: np.ndarray, mask: np.ndarray) -> tuple:
  """
  Masks a frame with the input mask.
  
  Parameters
  ----------
  frame : ndarray 
    The frame to be masked.
  model : ndarray
    Mask for the input frame.
    
  Returns
  -------
  result : ndarray
    Masked frame.
  non_zero_pixels : int
    Number of non-zero pixels on the mask.
  """
  non_zero_pixels = cv2.countNonZero(mask)
  result = cv2.bitwise_and(frame, frame, mask=mask)
  return result, non_zero_pixels
  
def merge_masks(masks: list, shape: tuple) -> np.ndarray:
  """
  Merges a list of mask into a single mask.
  
  Parameters
  ----------
  masks : list 
    List of mask to be merged.
  shape : tuple
    Height and width of each mask.
    
  Returns
  -------
  ndarray
    Merged mask.
  """
  merged_mask = np.zeros(shape, dtype=np.uint8)
  for mask in masks:
    merged_mask = cv2.bitwise_or(merged_mask, mask)
  return merged_mask

def nonZeroCounter(mask): #TODO: delete
  count = 0
  for i in range(len(mask)):
    for j in range(len(mask[i])):
      if mask[i][j] != 0:
        count += 1
  return count