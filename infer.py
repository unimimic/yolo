from libs.detector import ObjectDetector, ObjectDetectorOpenVINO
import cv2


# Example usage
detector = ObjectDetector('yolov8n.onnx', './yolov8n_openvino_model/metadata.yaml')
frame = cv2.imread('bus.jpg')
image, detections = detector.detect_objects(frame)

# Display the image with bounding boxes
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# # Example usage
# detector = ObjectDetectorOpenVINO('yolov8n_openvino_model/yolov8n.xml', './yolov8n_openvino_model/metadata.yaml')
# frame = cv2.imread('bus.jpg')
# image = detector.detect_objects(frame)

# # Display the image with detections
# cv2.imshow('Detections', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from ultralytics import YOLO

# # Load a YOLOv8n PyTorch model
# model = YOLO("yolov8n.pt")

# # Export the model
# model.export(format="onnx")  # creates 'yolov8n_openvino_model/'

# # Load the exported OpenVINO model
# ov_model = YOLO("yolov8n_openvino_model/")

# # Run inference
# results = ov_model("https://ultralytics.com/images/bus.jpg")