import numpy as np
import random
import time
import json

import torch
# torch.cuda.set_device(0)
from socketIO_client import SocketIO

import os
from local_yolo import FederatedLocal
from ultralytics.engine.model import Model
import logging
import argparse

# from models.yolo import Model
from ultralytics import YOLO
from uu import obj_to_pickle_string, pickle_string_to_obj

logging.getLogger('socketIO-client').setLevel(logging.WARNING)
random.seed(2023)
datestr = time.strftime('%m%d')
log_dir = os.path.join('logs', datestr)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)


class FLClient(object):

    def __init__(self):
        self.num = 1  # TODO set client num
        self.train_size = 828  # TODO set train size 878
        self.client_test_size = 208  # TODO set test size 1393
        self.nc = 1  # TODO set nc
        self.server_ip = '127.0.0.1'  # TODO set server ip
        self.port = 12345  # TODO set server port
        self.log_filename = f'logs/client_{self.num}.log'
        # logger
        self.logger = logging.getLogger("client")
        self.fh = logging.FileHandler(os.path.join(log_dir, os.path.basename(self.log_filename)))
        self.fh.setLevel(logging.INFO)

        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.ERROR)

        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(self.formatter)
        self.ch.setFormatter(self.formatter)

        self.logger.addHandler(self.fh)
        self.logger.addHandler(self.ch)
        self.local_model = FederatedLocal()
        # self.local_model = Model()
        self.sio = SocketIO(self.server_ip, self.port, None, {'timeout': 3600})
        self.register_handles()
        print("sent client_ready")
        self.sio.emit('client_ready')
        self.sio.wait()

    def register_handles(self):
        def on_request_update(*args):
            req = args[0]

            print("update requested")

            cur_round = req['current_round']
            self.logger.info("### Round {} ###".format(cur_round))

            if cur_round == 0:
                self.logger.info("received initial model")
            weights = req['current_weights']
            ema = req['ema']
            weights = pickle_string_to_obj(weights)
            ema = pickle_string_to_obj(ema)
            if req['STOP']:
                print("STOP")
                # device = torch.device('cuda', 0)
                # model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')  # 默认选n
                # # model = Model('cfg/training/yolov7.yaml', ch=3, nc=self.nc, anchors=0)
                # model.load_state_dict(weights.float().state_dict())
                torch.save(weights, f'client_{self.num}_last.pt')
                exit(0)
            my_weights, my_ema, results = self.local_model.train(weights, ema, self.num)
            torch.cuda.empty_cache()
            # self.datanum += 1

            resp = {
                'round_number': cur_round,
                'weights': obj_to_pickle_string(my_weights),
                'ema': obj_to_pickle_string(my_ema),
                'train_size': self.train_size,
                'client_test_loss': results[6],
                'client_test_map': results[2],
                'client_test_recall': results[1],
                'client_test_size': self.client_test_size,
            }

            print("Emit client_update")
            self.sio.emit('client_update', resp)
            self.logger.info("sent trained model to server")
            print("Emited...")

        self.sio.on('train_next_round_or_stop', on_request_update)


if __name__ == "__main__":

    try:
        FLClient()
    except ConnectionError:
        print('The server is down. Try again later.')
