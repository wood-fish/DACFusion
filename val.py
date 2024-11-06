from ultralytics import YOLO
from ultralytics import YOLOv10

# Load a model
#model = YOLO('yolov10m.yaml')  # build a new model from YAML
# model = YOLOv10('/home/qjc/Project/yolov10-main/runs/detect/lsknet_mul_250ep_twok_x*attn/weights/best.pt')
model = YOLO('/home/qjc/Project/yolov10-main/runs/detect/VEDAI_yolov8/weights/best.pt')
# model = YOLOv10('/home/qjc/Project/yolov10-main/runs/detect/llvipmul_250ep_twok2/weights/best.pt')
# load a pretrained model (recommended for training)
# model = YOLOv10('/home/qjc/Project/yolov10-main/runs/detect/FLIR_align2/weights/best.pt')  # load a pretrained model (recommended for training)
#model = YOLOv10('yolov10m.yaml').load('yolov10m.pt')  # build from YAML and transfer weights

# Train the model
# model.val(data='SRvedai.yaml',batch=2,imgsz=1024,workers=8,device=0,save_json=False)
# model.val(data='FLIR.yaml',batch=8,imgsz=640,workers=8,device=0,save_json=False)
# model.val(data='LLvip.yaml',batch=1,imgsz=1024,workers=1,device=0,save_json=False)
# model.predict(source='/home/qjc/DataSet/DATA/VEDAI_1024/images/00000485_ir.png',batch=1, device=0, workers=1, imgsz=1024,save=True, show_boxes=True,show_conf=False,show_labels=False)
# model.predict(source='/home/qjc/DataSet/FLIR-align/visible/train/FLIR_00003_PreviewData_day.jpg', device=0, imgsz=640,save=False, show_boxes=True,show_conf=True,show_labels=True,conf=0.01)
# results = model.predict(source='/home/qjc/DataSet/LLvip/LLVIP/visible/train/080869.jpg', device=0, imgsz=1280,name="val_LLVIP",save=True, show_boxes=True,show_conf=False,show_labels=False)
model.predict(source='/home/qjc/DataSet/DATA/VEDAI_1024/images/00000350_co.png',batch=1, device=0, workers=1, imgsz=1024,name="VEDAI_val",cache=False,visualize=True)
# nohup yolo task=detect mode=train epochs=150 batch=32 device=0 workers=16 imgsz=640 patience=50