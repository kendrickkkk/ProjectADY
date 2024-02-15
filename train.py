from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml') 
model = YOLO('yolov8n.pt')  
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  

# Train the model
results = model.train(data='traffic-sign-yolov8.v1i.yolov8\data.yaml', epochs=50, imgsz=640)