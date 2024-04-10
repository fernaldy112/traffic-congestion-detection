from ultralytics.engine.results import Boxes

def ratio_pixel(vehicle_pixels: float, road_pixels: int):
  """
  Returns ratio between area of vehicles and area of the road
  
  Args:
    vehicle_pixels (float): Number of pixels in the image detected as vehicle
    road_pixels (int): Number of pixels in the image that is road area
    
  Returns:
    float: Division result between sum of area of all boxes and area of the road
  """
  return vehicle_pixels / road_pixels
  # TODO: if no boxes detected, sum_area = 0, division result is float 0.0 so it does not have item() method; recreate by masking non road area