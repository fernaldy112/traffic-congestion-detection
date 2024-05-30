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

def label_frame(frame: np.ndarray, 
                label: str, 
                org: tuple = (50, 50), 
                padding: int = 10,
                font: int = cv2.FONT_HERSHEY_SIMPLEX, 
                size: int = 1, 
                font_color: tuple = (255, 255, 255), 
                background_color: tuple = (0, 0, 0), 
                thickness: int = 1, 
                line_type: int = cv2.LINE_AA
                ) -> bool:
  """
  Annotate frame with label
  
  Parameters
  ----------
  frame : ndarray 
    The image to be annotated.
  label : str 
    Text to annotate the frame.
  org : tuple, default (50, 50)
    Coordinates of the label.
  padding : int, default 10 
    Padding of the background.
  font : int, default cv2.FONT_HERSHEY_SIMPLEX 
    Font type of the label.
  size : int, default 1
    Font scale factor.
  font_color : tuple, default (255, 255, 255) 
    Color of the label.
  background_color : tuple, default (0, 0, 0) 
    Color of the background.
  thickness : int, default 1 
    Thickness of the label.
  line_type : int, default cv2.LINE_AA 
    Type of the line.
    
  Returns
  -------
  boolean
    True.
  """
  x, y = org
  (w, h), _ = cv2.getTextSize(label, font, size, thickness)
  cv2.rectangle(frame, (x - padding, y - padding), (x + w + padding, y + h + padding), background_color, -1)
  cv2.putText(frame, label, (x, y + h + size - 1), font, size, font_color, thickness, line_type)
  return True