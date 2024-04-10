import cv2
import numpy as np

def manual_mask(frame, vertices):
  mask = np.zeros(frame.shape[:2], dtype=np.uint8)
  cv2.fillPoly(mask, [vertices], 255)
  non_zero_pixels = cv2.countNonZero(mask)
  # total_pixels = frame.shape[0] * frame.shape[1]
  # masked_out_pixels = total_pixels - non_zero_pixels
  result = cv2.bitwise_and(frame, frame, mask=mask)
  return result, non_zero_pixels

def get_yolov8_mask(frame, model):
  result = model(frame)[0]
  height, width, _ = frame.shape
  masks = [cv2.resize(mask.numpy()*255, (width, height)).astype(np.uint8) for mask in result.masks.data]
  merged_mask = merge_masks(masks, (height, width))
  return merged_mask

def yolov8_mask(frame, mask):
  non_zero_pixels = cv2.countNonZero(mask)
  result = cv2.bitwise_and(frame, frame, mask=mask)
  return result, non_zero_pixels
  
def merge_masks(masks: list, shape):
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