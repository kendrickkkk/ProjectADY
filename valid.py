from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  
model = YOLO('C:\ProjectADY/runs\detect/train\weights/best.pt')  


metrics = model.val()  
metrics.box.map    
metrics.box.map50  
metrics.box.map75  
metrics.box.maps  