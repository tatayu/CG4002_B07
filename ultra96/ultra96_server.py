from config import *
import socket
import threading
import base64
import random
from Crypto.Cipher import AES
from Crypto import Random

class Server(threading.Thread):
	def __init__:
		super(Server, self).__init__()

		self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		print('starting up on %s port %s' % ULTRA_ADDRESS)

	def decrypt_msg(self, cipher_text):
		decoded_message = base64.b64decode(cipher_text)
		iv = decoded_message[:16]
		secret_key = bytes(str(self.secret_key), encoding="utf8") 

		cipher = AES.new(secret_key, AES.MODE_CBC, iv)
		decrypted_message = cipher.decrypt(decoded_message[16:]).strip()
		decrypted_message = decrypted_message.decode('utf8')
		return decrypted_message

	def close(self):
		self.server.close()

	def run(self):
		self.server.bind(ULTRA_ADDRESS)
		self.server.listen()



def main():
	server = Server()
	server.run()

if __name__ == '__main__':
	main()
