from config import *
import socket
import threading
import base64
from Crypto.Cipher import AES
from Crypto import Random

class Client():
	def __init__(self):
		super(Client, self).__init__()

		self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.shutdown = threading.Event()

	def send_msg(self, msg):
		encrypted = self.encrypt_msg(msg)
		self.client.sendall(encrypted)

	def encrypt_msg(self, msg):
		#encrypt
		secret_key = SECRET_KEY
		msg += ' ' * (16 - (len(msg) % 16))
		iv = Random.new().read(AES.block_size)
		cipher = AES.new(str(secret_key).encode(), AES.MODE_CBC, iv)
		encoded = base64.b64encode(iv + cipher.encrypt(msg.encode()))
		return encoded

	def decrypt_msg(self, msg):
		#decrypt
		decoded_message = base64.b64decode(cipher_text)
		iv = decoded_message[:16]
		secret_key = bytes(str(self.secret_key), encoding="utf8") 

		cipher = AES.new(secret_key, AES.MODE_CBC, iv)
		decrypted_message = cipher.decrypt(decoded_message[16:]).strip()
		decrypted_message = decrypted_message.decode('utf8')

		decrypted_message = decrypted_message[decrypted_message.find('#'):]
		decrypted_message = bytes(decrypted_message[1:], 'utf8').decode('utf8')
		return decrypted_message

	def recv_msg(self):
		while True:
			data = self.client.recv(1024)

			if data:
				try:
					msg = data.decode("utf8")
					decrypted_message = self.decrypt_message(msg)
					print("{} :: {} :: {}".format(decrypted_message['position'],
                                                                  decrypted_message['action'], 
                                                                  decrypted_message['sync']))
				except Exception as e:
					print(e)

	def close(self):
		self.client.close()
		self.shutdown.set()

	def run(self):
		self.client.connect(EVAL_ADDRESS)
		msg = "#2 1 3|gun|1.87|"
		self.send_msg(msg)
		thread =threading.Thread(target=self.recv_msg())
		thread.start()
		self.close()

def main():
	client = Client()
	client.run()

if __name__ == '__main__':
	main()