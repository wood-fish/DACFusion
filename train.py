from ultralytics import YOLO
from ultralytics import YOLOv10

# Load a model
model = YOLO('DACFusion.yaml') # build a new model from YAML
# Train the model

# model.train(data='SRvedai.yaml',epochs=250 ,batch=4, device=0, workers=4, imgsz=1024,name="lsknet_rgb2_200ep",patience=45)#conf=0.56,iou=0.89,dropout=0.1
# model.train(data='SRvedai.yaml',epochs=10 ,batch=1, device=0, workers=1, imgsz=1024,name="try1",cache=False)
# model.train(data='LLvip.yaml',epochs=10 ,batch=1, device=0, workers=1, imgsz=1024,name="try",cache=False)
# model.train(data='LLvip.yaml',epochs=250 ,batch=2, device=0, workers=8, imgsz=1280,name="LLVIP_v8_try",patience=35,cache=False)
# model.train(data='LLvip_sample.yaml',epochs=250 ,batch=2, device=0, workers=4, imgsz=1280,name="LLVIP_smallSample",patience=36,cache=False)
# model.train(data='SRvedai.yaml',epochs=250 ,batch=4, device=0, workers=8, imgsz=1024,name="VEDAI_nc9_v8_pre",patience=35,cache=False)#conf=0.56,iou=0.89,dropout=0.1
# model.train(data='SRvedai_512.yaml',epochs=250 ,batch=4, device=0, workers=16, imgsz=1024,name="VEDAI_nc9_512",patience=50,cache=False)#conf=0.56,iou=0.89,dropout=0.1
# model.train(data='data2.yaml',epochs=300 ,batch=8, device=0, workers=8, imgsz=640,name="DroneVenicle",patience=50)#conf=0.56,iou=0.89,dropout=0.1
model.train(data='FLIR.yaml',epochs=250 ,batch=2, device=0, workers=4, imgsz=640,name="FLIR_align_640size",patience=35)#conf=0.56,iou=0.89,dropout=0.1
# model.train(data='FLIR.yaml',epochs=10 ,batch=1, device=0, workers=1, imgsz=640,name="FLIR_align_try",patience=35,mosaic=0.1)#conf=0.56,iou=0.89,dropout=0.1
# model.train(data='KAIST.yaml',epochs=20 ,batch=1, device=0, workers=1, imgsz=640,name="KAIST_try",patience=35)#conf=0.56,iou=0.89,dropout=0.1
#model.train(data='ultralytics/cfg/datasets/DOTAv1.yaml',epochs=100 ,batch=8, device=0, workers=8, imgsz=640,patience=30,name="lsknet_DOTA1")#,cache=True)#conf=0.56,iou=0.89,dropout=0.1
#model.train(data='ultralytics/cfg/datasets/DOTAv1.yaml',epochs=12 ,batch=8, device=0, workers=8, imgsz=640,patience=30,name="lsknet_DOTA1",lr0=0.0002,lrf=0.0002,weight_decay=0.05)
#nohup yolo task=detect mode=train epochs=150 batch=32 device=0 workers=16 imgsz=640 patience=50
# model.train(resume=True)