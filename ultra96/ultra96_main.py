from config import *
from ml_stub import MLStub
from ultra96_server import Server
from ultra96_client import Client
from hardware_accelerator import HardwareAccelerator
from results import feature_extraction, knn_predict
from cnn import cnn_predict, predict
from mlp import mlp_predict
import numpy as np
import socket
import threading
import base64
import random
import queue
import time
import statistics
from Crypto.Cipher import AES
from Crypto import Random

import warnings

warnings.filterwarnings("ignore")

class Ultra96Main(threading.Thread):
        def __init__(self):
                super(Ultra96Main, self).__init__()

                self.msg_queue = queue.Queue()
                # Use MLStub() for local testing
                self.action_dict = ["dab", "elbowkick", "gun", "hair", "listen", "pointhigh", "sidepump", "wipetable"]
                #self.ml = MLStub()
                self.ml = HardwareAccelerator()

                self.all_connected = threading.Event()
                self.lastPredictTime = 0
                self.isSent = False
                self.isMoved = False

                self.dance_data = {}
                self.movements = ["-", "-", "-"]
                self.begin_movement_time = 0
                self.dancer_positions = [1, 2, 3]
                self.dancer_timestamps = {}
                self.logout_timestamps = {}
                self.server = Server(ultra96=self)
                self.client = Client(ultra96=self)

        def run(self):
                self.server.setDaemon(True)
                self.server.run()

                while not (self.all_connected.is_set()):
                        time.sleep(1)

                self.client.run()

        def init_dancer(self, dancer_id):
                self.dance_data[dancer_id] = queue.Queue(maxsize=300)
                self.logout_timestamps[dancer_id] = 0
                print("[INIT] Initialised dancer! Dancer ID: {}".format(dancer_id))

        def set_dancer_positions(self, msg):
                if self.server.isPredict:
                    self.isSent = True
                self.clear_data()
                split_msg = msg.split()
                self.dancer_positions[0], self.dancer_positions[1], self.dancer_positions[2] = split_msg[:3]
                to_print = f"[POS] New Dancer Positions: {self.dancer_positions[0]}|{self.dancer_positions[1]}|{self.dancer_positions[2]}"
                print(to_print)
                #self.clear_data()
                #if self.server.isPredict:
                #    self.isSent = True
                self.isMoved = False
                self.server.isPredict = False
                self.lastPredictTime = int(round(time.time() * 1000))
                    
        def send_logout(self):
            sync_delay = max(self.logout_timestamps.values()) - min(i for i in self.logout_timestamps.values() if i > 0) 

            self.client.send_prediction(self.dancer_positions[0], self.dancer_positions[1], self.dancer_positions[2], "logout", sync_delay)
 

        def update_dancer_positions(self):
            move_list = "Invalid Movements"
            #print(self.movements)
            if self.isMoved:
                return
            if (self.movements.count("-") == 1):
                if (self.movements[int(self.dancer_positions[0]) - 1] == "-"):
                    self.dancer_positions[1], self.dancer_positions[2] = self.dancer_positions[2], self.dancer_positions[1]
                    move_list = "N | R | L"
                elif (self.movements[int(self.dancer_positions[1]) - 1] == "-"):
                    self.dancer_positions[0], self.dancer_positions[2] = self.dancer_positions[2], self.dancer_positions[0]
                    move_list = "R | N | L"
                elif (self.movements[int(self.dancer_positions[2]) - 1] == "-"):
                    self.dancer_positions[0], self.dancer_positions[1] = self.dancer_positions[1], self.dancer_positions[0]
                    move_list = "R | L | N"
            elif ("-" not in self.movements):
                if (self.movements[int(self.dancer_positions[0]) - 1] == "L"):
                    self.movements[int(self.dancer_positions[0]) - 1] = "R"
                if (self.movements[int(self.dancer_positions[2]) - 1] == "R"):
                    self.movements[int(self.dancer_positions[2]) - 1] = "L"
                if (self.movements[int(self.dancer_positions[1]) - 1] == "R"):
                        temp = self.dancer_positions[2]
                        self.dancer_positions[2] , self.dancer_positions[1]= self.dancer_positions[1], self.dancer_positions[0]
                        self.dancer_positions[0] = temp
                        move_list = "R | R | L"
                else:
                        temp = self.dancer_positions[0]
                        self.dancer_positions[1], self.dancer_positions[0] = self.dancer_positions[2], self.dancer_positions[1]
                        self.dancer_positions[2] = temp
                        move_list = "R | L | L"
            check_pos = self.movements
            self.movements = ["-", "-", "-"]
            if (move_list == "Invalid Movements"):
                to_print = f"[MOV] {move_list} | {check_pos}"
            else:
                self.isMoved = True
                to_print = f"[MOV] {check_pos}"
                # sd = 1010
                # self.client.send_prediction(self.dancer_positions[0], self.dancer_positions[1], self.dancer_positions[2], "gun", sd)
   
            print(to_print)
            to_print = f"[POS] New Dancer Positions: {self.dancer_positions[0]}|{self.dancer_positions[1]}|{self.dancer_positions[2]}"
            print(to_print)
            # self.clear_data()

        def pass_dance_data(self, dancer_id, data):
                #with self.lock:
                recv_time  = int(round(time.time() *1000))
                if (not self.server.isPredict and recv_time - self.lastPredictTime > 3000):
                    #print(f"[DATA] Dancer {dancer_id} qsize: {self.dance_data[dancer_id].qsize()}")
                    self.dance_data[dancer_id].put(data)
                    self.predict_move()

        def clear_data(self):
            print("[CLEARING] ...")
            with self.dance_data["1"].mutex:
                self.dance_data["1"].queue.clear()
                self.dance_data["1"].unfinished_tasks = 0
            with self.dance_data["2"].mutex:
                self.dance_data["2"].queue.clear()
                self.dance_data["2"].unfinished_task = 0
            with self.dance_data["3"].mutex:
                self.dance_data["3"].queue.clear()
                self.dance_data["3"].unfinished_task = 0
            self.dancer_timestamps.clear()

        
        def predict_move(self): 
                action = ""
                size_1 = self.dance_data["1"].qsize()
                size_2 = self.dance_data["2"].qsize()
                size_3 = self.dance_data["3"].qsize()

                qsize_list = [size_1 , size_2, size_3]
                diff = max(qsize_list) - min(i for i in qsize_list if i > 0)
                if (qsize_list.count(0) < 2) and (diff >= 250):
                    self.clear_data()
                #if (any(dancer_data.qsize() >= 280 for dancer_data in self.dance_data.values())) and not self.server.isPredict:
                if ((min(i for i in qsize_list if i > 0) >= 280) and not self.server.isPredict):
                    self.server.isPredict = True
                    print("[PREDICTING] ...")
                    collated_data = []

                    dance = None
                    try:
                            values, dance, returnType = predict(self.dancer_positions, self.dance_data["1"], self.dance_data["2"], self.dance_data["3"])
                            # values, dance, returnType = mlp_predict(self.dancer_positions, self.dance_data["1"], self.dance_data["2"], self.dance_data["3"]) 
                            # _, dance_check, _ = knn_predict(self.dancer_positions, self.dance_data["1"], self.dance_data["2"], self.dance_data["3"])
                            # print("Dance check:", dance_check)
                    except Exception as e:
                            print(f"Exception found: {str(e)}")
                            pass
                    # if dance_check == 0:
                    #    dance = dance_check
                   # print(f"q1: {self.dance_data['1'].qsize()}, q2: {self.dance_data['2'].qsize()}, q3: {self.dance_data['3'].qsize()}") 
                    print(dance)
                    if (dance is not None and (not self.isSent)):
                        sync_delay = max(self.dancer_timestamps.values()) - min(i for i in list(self.dancer_timestamps.values()) if i > 0)

                        self.client.send_prediction(self.dancer_positions[0], self.dancer_positions[1], self.dancer_positions[2], self.action_dict[dance], sync_delay)
                        self.movements = ["-", "-", "-"]
                        self.isSent = True
                    else:
                        print("Dance accuracy below confidence level. Skipping...")

                    self.clear_data()
                    self.server.start_clock_sync()

                    self.server.isPredict = False
                    self.isSent = False
                    print("[START] Start dancing again!!")

def main():
        ultra96Main = Ultra96Main()
        ultra96Main.run()

if __name__ == '__main__':
        main()
                        

