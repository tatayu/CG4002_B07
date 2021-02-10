from config import *
import sshtunnel
import socket
import threading
from Crypto.Cipher import AES
from Crypto import Random

class Client():
	def __init__(self):
		self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.one_way_trip_delay = 0
		self.server_offset = 0

	def start_tunnel(self, user, password, dest):
		tunnel1 = sshtunnel.open_tunnel(
			('sunfire.comp.nus.edu.sg', 22),
			remote_bind_address=dest,
			ssh_username=user,
			ssh_password=password,
			block_on_close=False
		)
		tunnel1.start()
		print('[Tunnel Opened] Sunfire tunnel opened: ' + str(tunnel1.local_bind_port))

		tunnel2 = sshtunnel.open_tunnel(
			ssh_address_or_host=(
			'localhost', tunnel1.local_bind_port),
			remote_bind_address=('127.0.0.1', 8081),
			ssh_username='xilinx',
			ssh_password='xilinx',
			local_bind_address=('127.0.0.1', 8081),
			block_on_close=False
		)
		tunnel2.start()
		print('[Tunnel Opened] Xilinx tunnel opened')

	def encrypt_msg(self, msg):
		secret_key = SECRET_KEY
		msg += ' ' * (16 - (len(msg) % 16))
		iv = Random.new().read(AES.block_size)
		cipher = AES.new(str(secret_key).encode(), AES.MODE_CBC, iv)
		encoded = base64.b64encode(iv + cipher.encrypt(msg.encode()))
		return encoded

	def decrypt_msg(self, msg):
		decoded_message = base64.b64decode(cipher_text)
		iv = decoded_message[:16]
		secret_key = bytes(str(self.secret_key), encoding="utf8") 

		cipher = AES.new(secret_key, AES.MODE_CBC, iv)
		decrypted_message = cipher.decrypt(decoded_message[16:]).strip()
		decrypted_message = decrypted_message.decode('utf8')

		return decrypted_message

	def send_msg(self, msg):
		encrypted = self.encrypt_msg(msg)
		self.client.send_all(encrypted)

	def poll_for_start(self):
		while True:
			try:
				data = self.client.recv()
				msg = self.decrypt_msg(data)
				if ("[S]" in msg):
					break
			except Exception as e:
				print(e)

	def start_up(self):
		self.poll_for_start()

		#start the thread

	def run(self):
		self.start_tunnel(SUNFIRE_USERNAME, SUNFIRE_PASSWORD, ULTRA_ADDRESS)
		
		try:
			self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self.client.settimeout(1)
			self.client.connect(HOST_ADDRESS)
			print("[ULTRA96 CONNECTED] You are connected to Ultra96")
			self.start_up()
			time.sleep(1)
		except Exception as e:
			pass
                
if __name__ == '__main__':
    dancer_client = Client()
    dancer_client.run()