from ultralytics import YOLO
from ultralytics import YOLOv10

# Load a model
#model = YOLO('yolov10m.yaml')  # build a new model from YAML
model = YOLOv10('/home/qjc/Project/yolov10-main/runs/detect/train13/weights/best.pt')  # load a pretrained model (recommended for training)
#model = YOLOv10('yolov10m.yaml').load('yolov10m.pt')  # build from YAML and transfer weights

# Train the model
model.predict(source='/home/qjc/DataSet/DOTAv1/DOTAv1/images/test',name='v10_DOTAv1')

#nohup yolo task=detect mode=train epochs=150 batch=32 device=0 workers=16 imgsz=640 patience=50