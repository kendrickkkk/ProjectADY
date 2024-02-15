from ultralytics import YOLO
model = YOLO('C:\ProjectADY/runs\detect/train4\weights/best.pt')

model.predict('OIP.jpg',save=True)