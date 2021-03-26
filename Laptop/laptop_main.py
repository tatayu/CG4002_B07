from config import *
import sshtunnel
import socket
import threading
import base64
import time
import queue
from laptop_client import Client
from Crypto.Cipher import AES
from Crypto import Random

class LaptopMain(threading.Thread):
	def __init__(self):
		super(LaptopMain, self).__init__()

		self.data_queue = queue.Queue()
		self.client = Client(laptop=self)
		self.movement = "-"

	def run(self):
		#self.client.setDaemon(True)
		self.client.run()

	def insert(self, data):
		self.data_queue.put(data)

	def collect(self, data):
		return self.data_queue.get()
	
	def position(self, movement):
		self.movement = movement