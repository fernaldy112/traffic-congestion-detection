def ratio_pixel(vehicle_pixels: float, road_pixels: int) -> float:
  """
  Returns ratio between area of vehicles and area of the road.
  
  Parameters
  ----------
  vehicle_pixels : float
    Number of pixels in the image detected as vehicle.
  road_pixels : int
    Number of pixels in the image that is road area.
    
  Returns
  -------
  float
    Division result between sum of area of all boxes and area of the road.
  """
  if road_pixels == 0:
    return 0
  return vehicle_pixels / road_pixels