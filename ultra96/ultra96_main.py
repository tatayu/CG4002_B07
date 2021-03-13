from config import *
from ml_stub import MLStub
from ultra96_server import Server
from ultra96_client import Client
from hardware_accelerator import HardwareAccelerator
import numpy as np
import socket
import threading
import base64
import random
import queue
from Crypto.Cipher import AES
from Crypto import Random

class Ultra96Main(threading.Thread):
	def __init__(self):
		super(Ultra96Main, self).__init__()

		self.msg_queue = queue.Queue()
		# Use MLStub() for local testing
		#self.ml = MLStub()
		self.ml = HardwareAccelerator()
		self.server = Server(ultra96=self)
		self.client = Client(ultra96=self)
		self.all_connected = threading.Event()
		#self.lock = threading.Lock()

		self.dance_data = {}
		self.dancer_positions = [1, 2, 3]

	def run(self):
		self.server.setDaemon(True)
		self.server.run()

		while not (self.all_connected.is_set()):
			time.sleep(1)

		self.client.run()

	def init_dancer(self, dancer_id):
		self.dance_data[dancer_id] = queue.Queue()
		print("[INIT] Initialised dancer! Dancer ID: {}".format(dancer_id))

	def set_dancer_positions(self, msg):
		split_msg = msg.split()
		self.dancer_positions[0], self.dancer_positions[1], self.dancer_positions[2] = split_msg[:3]
		to_print = f"[POS] New Dancer Positions: {self.dancer_positions[0]}|{self.dancer_positions[1]}|{self.dancer_positions[2]}"
		print(to_print)

	def pass_dance_data(self, dancer_id, data):
		#with self.lock:
		self.dance_data[dancer_id].put(data)
		self.predict_move()

	def predict_move(self):
		#placeholder sync delay
		sync = 1.23

		for dancer_id in self.dance_data.keys():
			action = self.ml.predict(np.asarray(self.dance_data[dancer_id].get()))

		self.client.send_prediction(self.dancer_positions[0], self.dancer_positions[1], self.dancer_positions[2], action, sync)

def main():
	ultra96Main = Ultra96Main()
	ultra96Main.run()

if __name__ == '__main__':
	main()