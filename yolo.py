from ultralytics import YOLO

# Load a model
model = YOLO("/keypoints/yolo11s-pose.pt")  # load an official model
model = YOLO("/coin/yolo11s.pt")  # load a custom model

results = model.track("https://raw.githubusercontent.com/Fatikhaaa/capstoneProject/main/images/baby_image/baby_1.jpeg")