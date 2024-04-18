import cv2
import numpy as np

def resize(frame: np.ndarray, new_width: int = 840) -> np.ndarray:
  """
  Resizes the frame while maintaining its original aspect ratio.
  
  Parameters
  ----------
  frame : ndarray 
    The image to be resized.
  new_width : int, default 840
    New width of the frame.
    
  Returns
  -------
  ndarray
    Resized image.
  """
  ratio = new_width / frame.shape[1]
  new_height = int(frame.shape[0] * ratio)
  return cv2.resize(frame, (new_width, new_height))

def to_grayscale(frame: np.ndarray) -> np.ndarray:
  """
  Convert the frame into a grayscale image.
  
  Parameters
  ----------
  frame : ndarray 
    The image to be converted into grayscale image.
    
  Returns
  -------
  ndarray
    Grayscale image.
  """
  return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)