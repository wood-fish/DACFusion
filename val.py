from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/VEDAI/weights/best.pt') #Fill in the trained model path here

# Train the model
model.val(data='SRvedai.yaml',batch=4,imgsz=1024,workers=8,device=0,save_json=False)
# model.val(data='LLvip.yaml',batch=2,imgsz=1024,workers=1,device=0,save_json=False)