import numpy as np
from skimage.feature import graycomatrix, graycoprops

def glcm_properties(frame_gray: np.ndarray, properties: list = ["contrast"], distances: list = [1], angles: list = [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric: bool = True, normed: bool = True) -> np.ndarray:
  """
  Returns an array of mean value of each texture property from the grayscale input frame.
  
  Parameters
  ----------
  frame_gray : ndarray
    A grayscale image to be analyzed.
  properties : list, default ["contrast"]
    List of GLCM properties to be analyzed from the image. Valid properties are 
    contrast, dissimilarity, homogeneity, ASM, energy, and correlation.
  distances : list, default [1]
    List of pixel pair distances to be analyzed from the image.
  angles : list, default [0, np.pi/4, np.pi/2, 3*np.pi/4]
    List of pixel pair angles to be analyzed from the image.
  symmetric : bool, default True
    If True, the GLCM matrix is symmetric.
  normed : bool, default True
    If True, the GLCM matrix values are normalized.
  
  Returns
  -------
  ndarray
    i-th element is the mean value of i-th texture property. Mean value is calculated by considering every distance and angle.
  """
  glcm_matrix = graycomatrix(frame_gray, distances = distances, angles = angles, levels = 256, symmetric = symmetric, normed = normed) # glcm_matrix has shape of (256, 256, |distances|, |angles|)
  means = []
  for property in properties:
    glcm_property_matrix = graycoprops(glcm_matrix, property) # glcm_property_matrix[d][a] is property value for d-th distance and a-th angle
    mean = np.mean(glcm_property_matrix)
    means = np.append(means, mean)
  return means