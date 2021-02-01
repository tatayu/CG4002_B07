from config import *
import sshtunnel

class Client():
	def __init__(self, ip_addr):
		self.ip_addr = ip_addr

	def start_tunnel(self, user, password):
		tunnel1 = sshtunnel.open_tunnel(
			('sunfire.comp.nus.edu.sg', 22),
			remote_bind_address=('137.132.86.230', 22),
			ssh_username=user,
			ssh_password=password,
			block_on_close=False
		)
		tunnel1.start()
		print('[Tunnel Opened] Sunfire tunnel opened' + str(tunnel1.local_bind_port))

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

	def run(self):
		self.start_tunnel(SUNFIRE_USERNAME, SUNFIRE_PASSWORD)
		
		# Continuously connect to the Ultra96
        while True:
            try:
                self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client.settimeout(1)
                self.client.connect(self.ip_addr)
                print(f"[ULTRA96 CONNECTED] You are connected to Ultra96")
                self.procedure()
                time.sleep(1)
            except ConnectionRefusedError:
                self.is_start.clear()
                time.sleep(1)
                print("[TRYING] Connection refused!")
            except Exception as e:
                pass
                
if __name__ == '__main__':
    dancer_client = Client(HOST_ADDR)
    dancer_client.run()