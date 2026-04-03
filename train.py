from ultralytics import YOLO

# Load a model
model = YOLO('/home/hjy/workspace/DACFusion/DACnet_inn_v8.yaml') # build a new model from YAML

# Train the model

model.train(data='SRvedai.yaml',epochs=250 ,batch=4, device=0, workers=8, imgsz=1024,name="VEDAI",patience=35,cache=False)
# model.train(data='LLvip.yaml',epochs=1000 ,batch=4, device=1, workers=4, imgsz=1280,name="LLVIP",patience=35,cache=False)

# model.train(resume=True)