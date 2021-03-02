from config import *
from ultra96_server import Server
from ultra96_client import Client
import socket
import threading
import base64
import random
from Crypto.Cipher import AES
from Crypto import Random

class Ultra96Main(threading.Thread):
	def __init__(self):
		super(Ultra96Main, self).__init__()
		self.server = Server(ultra96=self)
		self.client = Client(ultra96=self)
		self.all_connected = threading.Event()

		self.dance_data = {}
		self.dancer_positions = [1, 2, 3]

	def run(self):
		self.server.setDaemon(True)
		self.server.run()

		while not (self.all_connected.is_set()):
			time.sleep(1)

		self.client.run()

	def init_dancer(self, dancer_id):
		self.dance_data[dancer_id] = []
		print("[INIT] Initialised dancer! Dancer ID: {}".format(dancer_id))

	def set_dancer_positions(self, msg):
		split_msg = msg.split()
		self.dancer_positions[0], self.dancer_positions[1], self.dancer_positions[2] = split_msg[:3]
		to_print = f"[POS] New Dancer Positions: {self.dancer_positions[0]}|{self.dancer_positions[1]}|{self.dancer_positions[2]}"
		print(to_print)

def main():
	ultra96Main = Ultra96Main()
	ultra96Main.run()

if __name__ == '__main__':
	main()