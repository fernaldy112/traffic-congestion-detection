import ultralytics
import supervision as sv
import cv2
import numpy as np
import joblib
import argparse

from typing import Union
from tensorflow.keras.models import load_model, Sequential
from sklearn.preprocessing import StandardScaler

from utils.format import xyxy_to_vertices
from utils.core import resize, to_grayscale, label_frame
from utils.mask import yolov8_mask, get_yolov8_mask
from utils.flow import count_object
from utils.occupancy import ratio_pixel
from utils.density import glcm_properties
from utils.velocity import lk_optical_flow, visualize_lk

def process_video(filename: str, 
                  vehicle_detection_model: ultralytics.YOLO, 
                  road_segmentation_model: ultralytics.YOLO, 
                  classification_model: Sequential, 
                  scaler: StandardScaler, 
                  device: Union[int, str]):
  """
  Process a video frame by frame to detect traffic congestion.
  
  Parameters
  ----------
  filename : str 
    Name of the video file.
  vehicle_detection_model : YOLO 
    Model for vehicle detection.
  road_segmentation_model : YOLO 
    Model for road segmentation.
  classification_model : Sequential
    Model for traffic status classification based on features analyzed.
  scaler : StandardScaler
    Model for features scaling before classification.
  device : {0, "cpu"}
    Device for vehicle detection and road segmentation.
  """
  SOURCE_VIDEO_PATH = f"./videos/{filename}.mp4"

  cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

  FRAME_INTERVAL = int(cap.get(cv2.CAP_PROP_FPS))
  CONFIDENCE_THRESHOLD = 0.0
  OBJECT_RATIO_THRESHOLD = 0.5

  success, previousFrame = cap.read()
  mask = get_yolov8_mask(previousFrame, road_segmentation_model, device)
  previousFrame_masked_gray = to_grayscale(yolov8_mask(previousFrame, mask)[0])
  _, _ = cap.read()

  cv2.imshow("Mask", resize(mask, new_width=600))

  frame_count = 0

  while cap.isOpened():
    success, currentFrame = cap.read()
    
    if not success: # video ended
      break
    
    if frame_count % FRAME_INTERVAL == 0:
      currentFrame_masked, road_pixels = yolov8_mask(currentFrame, mask)
      currentFrame_masked_gray = to_grayscale(currentFrame_masked)
      
      results = vehicle_detection_model(currentFrame_masked, device=device)
      detections = sv.Detections.from_ultralytics(results[0])
      filtered_detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
      filtered_detections = filtered_detections[filtered_detections.area < OBJECT_RATIO_THRESHOLD * road_pixels]
      
      vehicle_pixels = filtered_detections.area.sum()
      vertices = xyxy_to_vertices(filtered_detections.xyxy, "midpoint")
      
      flow = count_object(filtered_detections)
      occupancy = ratio_pixel(vehicle_pixels, road_pixels)
      density = 1/glcm_properties(currentFrame_masked_gray, properties=["correlation"])[0]
      _, _, velocity = lk_optical_flow(currentFrame_masked_gray, previousFrame_masked_gray, vertices)
      
      X_scaled = scaler.transform(np.array([[flow, occupancy, density, velocity]]))
      prediction = np.argmax(classification_model.predict(X_scaled), axis=1)[0]
      if prediction == 0:
        prediction = "lancar"
      elif prediction == 1:
        prediction = "lambat"
      elif prediction == 2:
        prediction = "macet"
      
      bounding_box_annotator = sv.BoundingBoxAnnotator()
      annotated_frame = bounding_box_annotator.annotate(
        scene = currentFrame_masked.copy(),
        detections = filtered_detections
      )
      
      annotated_frame = resize(annotated_frame, new_width=600)
      cv2.imshow("Segmentation and Detection", annotated_frame)
      
      labeled_frame = resize(currentFrame, new_width=700)
      label_frame(labeled_frame, prediction)
      label_frame(labeled_frame, f"aliran: {flow}", org=(50, 100))
      label_frame(labeled_frame, f"okupansi: {round(float(occupancy), 3)}", org=(50, 150))
      label_frame(labeled_frame, f"kepadatan: {round(float(density), 3)}", org=(50, 200))
      label_frame(labeled_frame, f"kecepatan: {round(float(velocity), 3)}", org=(50, 250))
      cv2.imshow("Classification", labeled_frame)
      
    elif (frame_count + 2) % FRAME_INTERVAL == 0:
      previousFrame_masked_gray = to_grayscale(yolov8_mask(currentFrame, mask)[0])
    
    frame_count += 1
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break
    
  cap.release()
  cv2.destroyAllWindows()
  
def parse_arguments():
  """
  Parse command line arguments.
    
  Returns
  -------
  Namespace
    Files and device arguments.
  """
  parser = argparse.ArgumentParser(description="Traffic congestion detection on CCTV videos.")
  parser.add_argument("-f", "--files", type=str, nargs="+", required=True, help="Input file paths")
  parser.add_argument("-d", "--device", type=str, choices=["cpu", "gpu"], default="gpu", help="Device to use for object detection and segmentation.")
  args = parser.parse_args()
  if args.device == "gpu":
    args.device = 0
  return args
  
if __name__ == "__main__":
  args = parse_arguments()
  filenames = args.files
  device = args.device

  vehicle_detection_model = ultralytics.YOLO("./models/vehicle_detection.pt")
  vehicle_detection_model.fuse()

  road_segmentation_model = ultralytics.YOLO("./models/road_segmentation.pt")

  classification_model = load_model("./models/classification.h5")
  scaler = joblib.load("./models/scaler.pkl")
  
  print(type(classification_model))
  print(type(scaler))
  
  for filename in filenames:
    process_video(filename, vehicle_detection_model, road_segmentation_model, classification_model, scaler, device)
  