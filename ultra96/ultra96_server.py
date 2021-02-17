from config import *
import socket
import threading
import base64
import random
from Crypto.Cipher import AES
from Crypto import Random

class Server(threading.Thread):
	def __init__(self):
		super(Server, self).__init__()

		self.num_laptops = 1
		self.connected_laptops = []

		self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

	def send_msg(self, connection, msg):
		encrypted = self.encrypt_msg(msg)
		try:
			connnection.sendall(encrypted)
		except Exception as e:
			print(e)

	def encrypt_msg(self, msg):
		secret_key = SECRET_KEY
		msg += ' ' * (16 - (len(msg) % 16))
		iv = Random.new().read(AES.block_size)
		cipher = AES.new(str(secret_key).encode(), AES.MODE_CBC, iv)
		encoded = base64.b64encode(iv + cipher.encrypt(msg.encode()))
		return encoded

	def decrypt_msg(self, cipher_text):
		decoded_message = base64.b64decode(cipher_text)
		iv = decoded_message[:16]
		secret_key = b'0000000000000000'.encode()

		cipher = AES.new(secret_key, AES.MODE_CBC, iv)
		decrypted_message = cipher.decrypt(decoded_message[16:]).strip()
		decrypted_message = decrypted_message.decode('utf8')
		return decrypted_message

	def read_data(self, connection):
		data = connection.recv(1024)
		return data

	def handle_messages(self, connection, client_address):
		self.num_laptops -= 1

		while True:
			data = self.read_data(connection)
			if data:
				msg = self.decrypt_msg(data)
				msg = msg.strip()
				to_print = "[{}]: {}".format(client_address, msg)
				print(to_print)
				if ("10" in msg):
					break

	def start_dancing(self):
		for connection in self.connected_laptops:
			msg = "[S]"
			self.send_msg(connetion, msg)

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

		self.connected_laptops.append(connection)

		print ("[CONNECTED] All laptops connected!")
		self.start_dancing()

def main():
	server = Server()
	server.run()

if __name__ == '__main__':
	main()
