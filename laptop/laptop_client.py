from config import *
import sshtunnel
import socket
import threading
import base64
import time
from Crypto.Cipher import AES
from Crypto import Random

class Client(threading.Thread):
	def __init__(self, laptop):
		self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.laptop = laptop
		self.dancer_id = -999
		self.clock_offset = 0
		self.is_start = False

	def start_tunnel(self, user, password, ultra_address):
		tunnel1 = sshtunnel.open_tunnel(
			('sunfire.comp.nus.edu.sg', 22),
			remote_bind_address=ultra_address,
			ssh_username=user,
			ssh_password=password,
			block_on_close=False
		)
		tunnel1.start()
		print('[Tunnel Opened] Sunfire tunnel opened: ' + str(tunnel1.local_bind_port))

		tunnel2 = sshtunnel.open_tunnel(
			ssh_address_or_host=(
			'127.0.0.1', tunnel1.local_bind_port),
			ssh_username='xilinx',
			ssh_password='xilinx',
			remote_bind_address=('127.0.0.1', 8081),
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
		decoded_message = base64.b64decode(msg)
		iv = decoded_message[:16]
		secret_key = bytes(str(SECRET_KEY), encoding="utf8")
		cipher = AES.new(secret_key, AES.MODE_CBC, iv)
		decrypted_message = cipher.decrypt(decoded_message[16:]).strip()
		decrypted_message = decrypted_message.decode('utf8')

		return decrypted_message

	def send_msg(self, msg):
		encrypted = self.encrypt_msg(msg)
		self.client.sendall(encrypted)

	def send_ready(self):
		msg = f"[S]|{self.dancer_id}"
		self.send_msg(msg)

	def send_data(self):
		print("In send_data thread!")
		while True:
			if not (self.laptop.data_queue.empty()):
				data = self.laptop.data_queue.get()

				data = data.replace(" ", "|")
				msg = f"[D]|{data}"
				print("Sending data!")
				self.send_msg(msg)

	def poll_for_start(self):
		while not self.is_start:
			time.sleep(0.5)

		data_thread = threading.Thread(target=self.send_data)
		data_thread.start()

	def clock_sync(self):
		t1 = int(round(time.time() * 1000))
		msg = f"[C]|{t1}"
		self.send_msg(msg)

		data = self.client.recv(256)
		t4 = int(round(time.time() * 1000))
		msg = self.decrypt_msg(data)
		split_msg = msg.split("|")
		t2 = float(split_msg[2])
		t3 = float(split_msg[3])

		rtt = ((t4 - t1) - (t3 - t2))
		#print(f"RTT: {rtt}")
		self.clock_offset = ((t2 - t1) + (t3 - t4))/2

		to_print = f"[CLOCK SYNC] Clock offset for Dancer {self.dancer_id}: {self.clock_offset}"
		print(to_print)

	def handle_ultra96_messages(self):
		while True:
			try:
				data = self.client.recv(256)
				if data:
					msg = self.decrypt_msg(data)
					msg = msg.strip()
					if ("[SC]" in msg):
						self.clock_sync()
					elif ("[S]" in msg):
						self.is_start = True
						print("[START] Start dancing!")
						self.clock_sync()
			except Exception as e:
				print(e)


	def start_up(self):
		self.send_ready()

		thread = threading.Thread(target=self.handle_ultra96_messages)
		thread.start()

		self.poll_for_start()


	def run(self):
		#self.start_tunnel(SUNFIRE_USERNAME, SUNFIRE_PASSWORD, ULTRA_ADDRESS)
		self.dancer_id = input("Input Dancer ID: ")
		
		try:
			self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self.client.connect(HOST_ADDRESS)
			print("[CONNECTED] Connected to Ultra96!")
			self.start_up()
			time.sleep(1)
		except Exception as e:
			print(e)
                
if __name__ == '__main__':
	pass
	dancer_client = Client()
	dancer_client.run()