import cv2

def resize(frame, new_width=840):
  ratio = new_width / frame.shape[1]
  new_height = int(frame.shape[0] * ratio)
  return cv2.resize(frame, (new_width, new_height))

def to_grayscale(frame):
  return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)