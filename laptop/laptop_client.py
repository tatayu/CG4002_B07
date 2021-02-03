from config import *
import sshtunnel
import socket
import threading

class Client():
	def __init__(self, ip_address):
		self.ip_address = ip_address

		self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.one_way_trip_delay = 0
		self.server_offset = 0

		self.is_start = threading.Event()

	def start_tunnel(self, user, password, target):
		tunnel1 = sshtunnel.open_tunnel(
			('sunfire.comp.nus.edu.sg', 22),
			remote_bind_address=target,
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

	def encrypt(self, msg):
		#encrypt

	def send_msg(self, msg):
		encrypted = self.encrypt(msg)
		self.client.send_all(encrypted)

	def start_up(self):
		print('In start_up function!')
		msg = "Start dancing!"
		send_msg(str(msg).encode())

	def run(self):
		self.start_tunnel(SUNFIRE_USERNAME, SUNFIRE_PASSWORD, TARGET_ADDRESS)
		
		try:
			self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self.client.settimeout(1)
			self.client.connect(self.ip_address)
			print("[ULTRA96 CONNECTED] You are connected to Ultra96")
			self.start_up()
			time.sleep(1)
		except Exception as e:
			pass
                
if __name__ == '__main__':
    dancer_client = Client(HOST_ADDRESS)
    dancer_client.run()