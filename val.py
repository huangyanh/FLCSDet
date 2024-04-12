import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/FLv8_15_48/weights/best.pt')  #runs/prune/yolov8n-seaship-groupsl-exp2-finetune/weights/best.pt
    model.val(data='dataset/data3C.yaml',
              split='test',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='v8_3C_15',
              )