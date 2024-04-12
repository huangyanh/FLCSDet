import sys
from copy import deepcopy

import torch
#
# from models.yolo import Model
import json
import random
import time
import uuid
import os
import numpy as np
from flask import *
from flask_socketio import *
import logging
import argparse

from ultralytics.utils.torch_utils import de_parallel

from ultralytics import YOLO
from ultralytics.nn import attempt_load_one_weight

# from utils.torch_utils import ModelEMA
from uu import pickle_string_to_obj, obj_to_pickle_string

datestr = time.strftime('%m%d')
timestr = time.strftime('%m%d%H%M')


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


class Aggregator(object):

    def __init__(self, logger, nc):
        self.logger = logger
        self.logger.info(self.get_model_description())
        self.nc = nc
        self.current_weights, self.current_emas = self.get_init_weights()

        # weights should be a ordered list of parameter
        # for stats
        self.train_losses = []
        self.avg_test_losses = []
        self.avg_test_maps = []
        self.avg_test_recalls = []

        # for convergence check
        self.prev_test_loss = None
        self.best_loss = None
        self.best_round = -1
        self.best_map = 0
        self.best_recall = 0

        self.training_start_time = int(round(time.time()))

    def get_init_weights(self):
        device = torch.device('cuda', 0)
        _, ckpt = attempt_load_one_weight('pretrain/yolov8n.pt')   #pretrain/yolov8n.pt
        weights = ckpt['model']  # yolov7.pt
        ema = ckpt['ema']
        del ckpt
        return weights, ema

    def update_weights(self, client_weights, emas, client_sizes):
        total_size = np.sum(client_sizes)

        new_weights = {}
        for key in client_weights[0].keys():
            print(client_weights[0][key].dtype)
            new_weight = torch.zeros_like(client_weights[0][key])
            print(new_weight.dtype)
            for i in range(len(client_weights)):
                new_weight += client_weights[i][key] * client_sizes[i]
            if client_weights[0][key].dtype == torch.float32:
                new_weight = new_weight / total_size
            else:
                new_weight = new_weight // total_size
            new_weights[key] = new_weight
            # new_weights[key] = 1-new_weight

        new_emas = {}
        for key in emas[0].keys():
            new_ema = torch.zeros_like(emas[0][key])
            for i in range(len(emas)):
                new_ema += emas[i][key] * client_sizes[i]
            if emas[0][key].dtype == torch.float32:
                new_ema = new_ema / total_size
            else:
                new_ema = new_ema // total_size
            new_emas[key] = new_ema

        # model = YOLO("yolov8.yaml")
        # model.model.load(new_weights)
        # emaa = ModelEMA(model)
        # emaa.ema.load_state_dict(new_emas)
        # model = YOLO("yolov5n.yaml")
        model = YOLO('ultralytics/cfg/models/v8/yolov8n-att.yaml')
        model.model.load_state_dict(new_weights)

        self.current_weights = deepcopy(de_parallel(model.model)).half()
        self.current_emas = new_emas
        del model

    def aggregate_loss_map_recall(self, client_losses, client_maps, client_recalls, client_sizes):
        total_size = sum(client_sizes)
        # weighted sum
        aggr_loss = sum(client_losses[i] / total_size * client_sizes[i]
                        for i in range(len(client_sizes)))
        aggr_maps = sum(client_maps[i] / total_size * client_sizes[i]
                        for i in range(len(client_sizes)))
        aggr_recalls = sum(client_recalls[i] / total_size * client_sizes[i]
                           for i in range(len(client_sizes)))
        return aggr_loss, aggr_maps, aggr_recalls

    def aggregate_loss_accuracy_recall(self, client_losses, client_maps, client_recalls, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_map, aggr_recall = self.aggregate_loss_map_recall(client_losses, client_maps, client_recalls,
                                                                          client_sizes)

        self.avg_test_losses += [[cur_round, cur_time, aggr_loss]]
        self.avg_test_maps += [[cur_round, cur_time, aggr_map]]
        self.avg_test_recalls += [[cur_round, cur_time, aggr_recall]]
        return aggr_loss, aggr_map, aggr_recall

    def get_stats(self):
        return {
            "train_loss": self.train_losses,
            "valid_loss": self.valid_losses,
            "train_accuracy": self.train_accuracies,
            "valid_accuracy": self.valid_accuracies,
            "train_recall": self.train_recalls,
            "valid_recall": self.valid_recalls
        }

    def get_model_description(self):
        return "Good morning, Sir."


# Federated Averaging algorithm with the server pulling from clients

class FLServer(object):
    def __init__(self, task_config_filename, host, port):
        self.task_config = load_json(task_config_filename)
        self.ready_client_sids = []

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, ping_timeout=360,
                                 ping_interval=3600,
                                 max_http_buffer_size=int(1e32))
        self.host = host
        self.port = port

        self.MIN_NUM_WORKERS = self.task_config["MIN_NUM_WORKERS"]
        self.MAX_NUM_ROUNDS = self.task_config["MAX_NUM_ROUNDS"]
        self.NUM_TOLERATE = self.task_config["NUM_TOLERATE"]
        self.NUM_CLIENTS_CONTACTED_PER_ROUND = self.task_config["NUM_CLIENTS_CONTACTED_PER_ROUND"]

        self.logger = logging.getLogger("aggregation")
        log_dir = os.path.join('logs', datestr, self.task_config['log_dir'])
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, '{}.log'.format(timestr)))
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.STOP = False

        self.logger.info(self.task_config)
        self.model_id = str(uuid.uuid4())

        self.aggregator = Aggregator(self.logger, self.task_config["nc"])

        self.current_round = -1
        self.current_round_client_updates = []
        self.eval_client_updates = []
        self.dis_id = 0

        self.register_handles()
        self.invalid_tolerate = 0

        @self.app.route('/stats')
        def status_page():
            return json.dumps(self.aggregator.get_stats())

    def register_handles(self):
        @self.socketio.on('client_ready')
        def handle_client_ready():
            print("client ready for training", request.sid)
            self.ready_client_sids.append(request.sid)
            if len(self.ready_client_sids) >= self.MIN_NUM_WORKERS:
                if self.current_round == -1:
                    print("start to federated learning.....")
                print(f"distribute the model to client{request.sid}")
                self.current_round += 1
                self.logger.info("### Round {} ###".format(self.current_round))
                self.current_round_client_updates = []
                self.eval_client_updates = []
                self.dis_id = 0
                self.distribute()
            else:
                print("waiting for more client worker.....")

        @self.socketio.on('client_update')
        def handle_client_update(data):
            self.logger.info("received client update of bytes: {}".format(sys.getsizeof(data)))
            self.logger.info("handle client_update {}".format(request.sid))

            self.current_round_client_updates += [data]
            self.current_round_client_updates[-1]['weights'] = pickle_string_to_obj(
                data['weights']).float().state_dict()
            self.current_round_client_updates[-1]['ema'] = pickle_string_to_obj(data['ema']).float().state_dict()
            if len(self.current_round_client_updates) == self.NUM_CLIENTS_CONTACTED_PER_ROUND:
                self.aggregator.update_weights(
                    [x['weights'] for x in self.current_round_client_updates],
                    [x['ema'] for x in self.current_round_client_updates],
                    [x['train_size'] for x in self.current_round_client_updates]
                )
                self.logger.info("=== aggregating ===")
                if 'client_test_loss' in self.current_round_client_updates[0]:
                    aggr_test_loss, aggr_test_map, aggr_test_recall = self.aggregator.aggregate_loss_accuracy_recall(
                        [x['client_test_loss'] for x in self.current_round_client_updates],
                        [x['client_test_map'] for x in self.current_round_client_updates],
                        [x['client_test_recall'] for x in self.current_round_client_updates],
                        [x['client_test_size'] for x in self.current_round_client_updates],
                        self.current_round
                    )
                    self.logger.info("=== server test ===")
                    self.logger.info("server_test_loss {}".format(aggr_test_loss))
                    self.logger.info("server_test_map {}".format(aggr_test_map))
                    self.logger.info("server_test_recall {}".format(aggr_test_recall))

                    if self.aggregator.best_map <= aggr_test_map:
                        self.aggregator.best_map = aggr_test_map
                        self.aggregator.best_loss = aggr_test_loss
                        self.aggregator.best_recall = aggr_test_recall
                        self.aggregator.best_round = self.current_round

                    if self.aggregator.prev_test_loss is not None and self.aggregator.prev_test_loss < aggr_test_loss:
                        self.invalid_tolerate = self.invalid_tolerate + 1
                    else:
                        self.invalid_tolerate = 0
                    self.aggregator.prev_test_loss = aggr_test_loss
                    if self.invalid_tolerate > self.NUM_TOLERATE > 0:
                        self.logger.info("converges! starting test phase..")
                        self.STOP = True
                if self.current_round >= self.MAX_NUM_ROUNDS:
                    self.logger.info("get to maximum step, stop...")
                    self.STOP = True
                # model = YOLO("yolov8.yaml")
                # model.model.load(self.aggregator.current_weights)
                torch.save(self.aggregator.current_weights, 'server_last.pt')
                if self.STOP:
                    self.logger.info("== done ==")
                    self.eval_client_updates = None  # special value, forbid evaling again
                    self.logger.info("Federated training finished ... ")
                    self.logger.info("best model at round {}".format(self.aggregator.best_round))
                    self.logger.info("get best test loss {}".format(self.aggregator.best_loss))
                    self.logger.info("get best map {}".format(self.aggregator.best_map))
                    self.logger.info("get best recall {}".format(self.aggregator.best_recall))

                    for sid in self.ready_client_sids:
                        emit('train_next_round_or_stop', {
                            'model_id': self.model_id,
                            'current_round': self.current_round,
                            'current_weights': obj_to_pickle_string(self.aggregator.current_weights),
                            'ema': obj_to_pickle_string(self.aggregator.current_emas),
                            'STOP': self.STOP
                        }, room=sid)
                        self.logger.info("sent aggregated model to client")

                    # torch.save(self.aggregator.current_weights.float().state_dict(), 'server_last.pt')
                    torch.save(self.aggregator.current_weights, 'server_last.pt')
                    exit(0)
                else:
                    self.logger.info("start to next round...")
                    self.current_round += 1
                    self.logger.info("### Round {} ###".format(self.current_round))
                    self.current_round_client_updates = []
                    self.eval_client_updates = []
            self.distribute()

    def distribute(self):

        emit('train_next_round_or_stop', {
            'model_id': self.model_id,
            'current_round': self.current_round,
            'current_weights': obj_to_pickle_string(self.aggregator.current_weights),
            'ema': obj_to_pickle_string(self.aggregator.current_emas),
            'STOP': self.STOP
        }, room=self.ready_client_sids[self.dis_id])
        self.dis_id = (self.dis_id + 1) % len(self.ready_client_sids)
        if self.current_round == 0:
            self.logger.info("sent initial model to client")
        else:
            self.logger.info("sent aggregated model to client")

    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="task config file")
    parser.add_argument("--port", type=int, required=True, help="server port")
    opt = parser.parse_args()
    print(opt)
    if not os.path.exists(opt.config_file):
        raise FileNotFoundError("{} dose not exist".format(opt.config_file))
    try:
        server = FLServer(opt.config_file, "127.0.0.1", opt.port)
        print("listening on 127.0.0.1:{}".format(str(opt.port)))
        server.start()
    except ConnectionError:
        print('Restart server fail.')
