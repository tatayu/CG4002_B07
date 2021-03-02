from config import *
import socket
import threading
import base64
import random
import time
from Crypto.Cipher import AES
from Crypto import Random

class Client(threading.Thread):
	def __init__(self, ultra96):
		super(Client, self).__init__()

		self.ultra96 = ultra96
		self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.shutdown = threading.Event()

	def send_msg(self, msg):
		encrypted = self.encrypt_msg(msg)
		self.client.sendall(encrypted)

	def encrypt_msg(self, msg):
		secret_key = SECRET_KEY
		msg += ' ' * (16 - (len(msg) % 16))
		iv = Random.new().read(AES.block_size)
		cipher = AES.new(str(secret_key).encode(), AES.MODE_CBC, iv)
		encoded = base64.b64encode(iv + cipher.encrypt(msg.encode("utf8")))
		return encoded

	def recv_msg(self):
		while True:
			data = self.client.recv(1024)

			if data:
				try:
					msg = data.decode("utf8")
					#print(msg)
					self.ultra96.set_dancer_positions(msg)
				except Exception as e:
					print(e)

	def send_prediction(self, p1, p2, p3, action, sync):
		prediction = f"#{p1} {p2} {p3}|{action}|{sync}"
		print(f"[SENDING] {prediction}")
		self.send_msg(prediction)

	def close(self):
		self.client.close()
		self.shutdown.set()

	def run(self):
		self.client.connect(EVAL_ADDRESS)
		thread = threading.Thread(target=self.recv_msg)
		thread.daemon = True
		thread.start()

		self.dummy_data()

	def dummy_data(self):
		time.sleep(65)

		position = [1, 2, 3]
		actions = ["gun", "sidepump", "hair"]
		sync = [1.23, 2.13, 3.12]

		for x in range(len(actions)):
			random.shuffle(actions)
			random.shuffle(position)
			random.shuffle(sync)

			self.send_prediction(position[0], position[1], position[2], actions[0], sync[0])
			time.sleep(5)

		self.send_prediction(position[0], position[1], position[2], 'logout', sync[0])
		quit()

def main():
	pass
	client = Client()
	client.run()

if __name__ == '__main__':
	main()