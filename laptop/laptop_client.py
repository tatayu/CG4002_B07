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

	def run(self):
		self.start_tunnel(SUNFIRE_USERNAME, SUNFIRE_PASSWORD)
		# Continuously connect to the Ultra96
        while True:
            try:
                self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client.settimeout(1)
                self.client.connect(self.ip_addr)
                print(f"[ULTRA96 CONNECTED] Dancer {self.dancer_id} is connected to Ultra96")
                self.procedure()
                time.sleep(1)
            except ConnectionRefusedError:
                self.is_start.clear()
                time.sleep(1)
                print("[TRYING] Why u no let me connect?? (┛ಠ_ಠ)┛彡┻━┻")
            except Exception as e:
                pass
                
if __name__ == '__main__':
    dancer_client = Client(HOST_ADDR)
    dancer_client.run()