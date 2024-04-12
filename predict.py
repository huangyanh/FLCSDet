import random

import torch

from ultralytics import YOLO
if __name__ == '__main__':

    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # select your model.pt path  server_last.pt C:\\zuomian\\Yolo\\ultralytics-FL\\runs\\train\\yolov5_2\\weights\\best.pt
    model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')
    a = torch.load('G:\\zuomian\\FL\\server_last\\C3\\v8_5.pt')     #runs/train/FLv5_23/weights/last.pt server_last.pt
    model.model.load(a)
    model.model.names = ['vessel']
    model.predict(source='dataset/Input2/client3',
                  imgsz=640,
                  save=True,
                  project='runs/predict',
                  name='v8_3C_5_',
                  #   visualize=True # visualize model feasures maps
                  )