from config import *
import socket
import threading
import base64
import random
import time
from Crypto.Cipher import AES
from Crypto import Random
import struct

class Server(threading.Thread):
	def __init__(self, ultra96):
		super(Server, self).__init__()

		self.ultra96 = ultra96
		self.num_laptops = 1
		self.connected_laptops = []
		self.dancers = []

		self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

	def send_msg(self, connection, msg):
		try:
			connection.sendall(msg.encode())
		except Exception as e:
			print(e)
			pass

	def handle_messages(self, connection, client_address):
		#one for each thread
		self.num_laptops -= 1
		self.dancer_id = -999


		while True:
			data = connection.recv(1024)
			if data:
				recv_time = int(round(time.time() * 1000))
				#msg = data.decode('utf8')

				if ("[C]" in msg):
					# Clock sync
					self.clock_sync(connection, msg, recv_time, dancer_id)
				elif ("[S]" in msg):
					# Record dancer details
					split_msg = msg.split("|")
					dancer_id = split_msg[1]
					self.ultra96.init_dancer(dancer_id)
				elif (b'D' in msg):
					unpackedData = struct.unpack('<I6h', msg[1:])
					str_msg = str(unpackedData)
					
					########################TODO##############################
					split_msg = msg.split("|")
					to_print = f"[DATA] Passing data from {dancer_id}: {msg}"
					print(to_print)
					self.ultra96.pass_dance_data(dancer_id, [float(s) for s in split_msg[1:]])



	def start_dancing(self):
		for connection in self.connected_laptops:
			msg = "[S]"
			self.send_msg(connection, msg)

	def clock_sync(self, connection, msg, recv_time, dancer_id):
		msg += f"|{recv_time}"
		send_time = int(round(time.time() * 1000))
		msg += f"|{send_time}"

		self.send_msg(connection, msg)

	def start_clock_sync(self):
		for connection in self.connected_laptops:
			#start clock sync
			msg = "[SC]"
			self.send_msg(connection, msg)

	def close(self):
		self.server.close()

	def run(self):
		self.server.bind(('127.0.0.1', 8081))
		print("[LISTENING] Waiting for laptops to connect!")
		self.server.listen(1)

		while self.num_laptops:
			try:
				connection, client_address =  self.server.accept()
				thread = threading.Thread(target=self.handle_messages, args=(connection, client_address))
				self.connected_laptops.append(connection)
				thread.start()
			except Exception as e:
				print(e)
				pass

		print ("[CONNECTED] All laptops connected!")
		self.ultra96.all_connected.set()
		self.start_dancing()
		#self.start_clock_sync()

def main():
	pass
	server = Server()
	server.run()

if __name__ == '__main__':
	main()
