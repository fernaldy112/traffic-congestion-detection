import ultralytics
import supervision as sv
import cv2
import numpy as np
import pandas as pd
import pickle

from utils.format import xyxy_to_vertices
from utils.core import resize, to_grayscale
from utils.mask import yolov8_mask, get_yolov8_mask
from utils.flow import count_object
from utils.occupancy import ratio_pixel
from utils.density import glcm_properties
from utils.velocity import lk_optical_flow

vehicle_detection_model = ultralytics.YOLO("./models/vehicle_detection.pt")
vehicle_detection_model.fuse()

road_segmentation_model = ultralytics.YOLO("./models/road_segmentation.pt")

with open("./models/lr.pkl", "rb") as file:
    classification_model = pickle.load(file)

FILENAME = "random"
SOURCE_VIDEO_PATH = f"./videos/{FILENAME}.mp4"
CONFIDENCE_THRESHOLD = 0.0
OBJECT_RATIO_THRESHOLD = 0.4

cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
FRAME_INTERVAL = int(cap.get(cv2.CAP_PROP_FPS))
success, previousFrame = cap.read()
mask = get_yolov8_mask(previousFrame, road_segmentation_model)
previousFrame_masked_gray = to_grayscale(yolov8_mask(previousFrame, mask)[0])

frame_count = 0

while cap.isOpened():
  success, currentFrame = cap.read()
  
  if not success: # video ended
    break
  
  if frame_count % FRAME_INTERVAL == 0:
    currentFrame_masked, road_pixels = yolov8_mask(currentFrame, mask)
    currentFrame_masked_gray = to_grayscale(currentFrame_masked)
    
    results = vehicle_detection_model(currentFrame_masked)
    detections = sv.Detections.from_ultralytics(results[0])
    filtered_detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
    filtered_detections = filtered_detections[filtered_detections.area < OBJECT_RATIO_THRESHOLD * road_pixels]
    
    vehicle_pixels = filtered_detections.area.sum()
    vertices = xyxy_to_vertices(filtered_detections.xyxy, "midpoint")
    
    n_object = count_object(filtered_detections)
    pixel_ratio = ratio_pixel(vehicle_pixels, road_pixels)
    contrast = glcm_properties(currentFrame_masked_gray, properties=["contrast"])[0]
    a, b, speed = lk_optical_flow(previousFrame_masked_gray, currentFrame_masked_gray, vertices)
    
    prediction = classification_model.predict(np.array([[n_object, pixel_ratio, contrast, speed]]))[0]
    prediction = "free" if prediction == 0 else "congested"
    
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    annotated_frame = bounding_box_annotator.annotate(
      scene = currentFrame_masked.copy(),
      detections = filtered_detections
    )
    
    annotated_frame = resize(annotated_frame)
    
    cv2.putText(annotated_frame, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    
  elif (frame_count + 1) % FRAME_INTERVAL == 0:
    previousFrame_masked_gray = to_grayscale(yolov8_mask(currentFrame, mask)[0])
  
  frame_count += 1
  
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break
  
cap.release()
cv2.destroyAllWindows()
