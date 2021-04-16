from config import *
import socket
import threading
import base64
import random
import time
import struct
from Crypto.Cipher import AES
from Crypto import Random

class Server(threading.Thread):
        def __init__(self, ultra96):
                super(Server, self).__init__()

                self.ultra96 = ultra96
                self.num_laptops = 3
                self.movement_time = 0
                self.connected_laptops = []
                self.dancers = []
                self.isPredict = False
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
                dancer_id = 1
                clock_offset = 0
                start_time = prev_time = int(round(time.time() * 1000))
                count = 0
                while True:
                        msg = None
                        curr_time = int(round(time.time()))
                        if (not self.isPredict) and (not self.movement_time == 0):
                            #print(f"curr: {curr_time}, start: {self.movement_time}")
                            diff = curr_time - self.movement_time
                            #print(f"diff: {diff}")
                            if (diff > 1):
                                a = curr_time - self.movement_time
                                #print(f"Time since start of movement: {a}")
                                self.movement_time = self.begin_movement_time = 0
                                self.ultra96.update_dancer_positions()

                        connection.settimeout(5.0)
                        try:
                            msg = connection.recv(17)
                        except socket.timeout:
                            print(f"[DISCONNECTED] Dancer {dancer_id}")
                            with self.ultra96.dance_data[dancer_id].mutex:
                                self.ultra96.dance_data[dancer_id].queue.clear()
                                self.ultra96.dance_data[dancer_id].unfinished_task = 0
                            #if (dancer_id in self.ultra96.dancer_timestamps.keys()):
                                #self.ultra96.dancer_timestamps.pop(dancer_id)
                        if msg:
                                recv_time = int(round(time.time() * 1000))                
                                if (b'D' in msg):
                                        if  (recv_time - prev_time < 2000):
                                                if (self.ultra96.dance_data[dancer_id].full()):
                                                        outdatedData = self.ultra96.dance_data[dancer_id].get()
                                                        
                                                try:
                                                        unpackedData = struct.unpack('<I6h', msg[1:])
                                                        
                                                        if not (dancer_id in self.ultra96.dancer_timestamps.keys()):
                                                            time_stamp = int(float(unpackedData[0]))
                                                            #print(f"time from Dancer {dancer_id}: {time_stamp}")    
                                                            self.ultra96.dancer_timestamps[dancer_id] = int(float(unpackedData[0])) - int(float(clock_offset))

                                                        to_print = f"[DATA] Passing data from {dancer_id}: {msg}"
                                                        if count % 80 == 0:
                                                            print(to_print)
                                                        count += 1
                                                        dance_data = list(unpackedData[1:])
                                                        dance_data = [int(i) for i in dance_data]
                                                
                                                        self.ultra96.pass_dance_data(dancer_id, dance_data)
                                                except Exception as e:
                                                        pass

                                        elif (recv_time - prev_time >= 5000):
                                            if (len(self.ultra96.dance_data) == 3):
                                                self.ultra96.clear_data()

                                        prev_time = recv_time
                        

                                else:
                                    try:
                                        msg = msg.decode('utf8')
                                    except UnicodeDecodeError as e:
                                        print(e)
                                        continue
                                    if ("[C]" in msg):
                                            # Clock sync
                                            #self.isPredict = True
                                            self.clock_sync(connection, msg, recv_time, dancer_id)
                                            #self.isPredict = False
                                    elif ("[S]" in msg):
                                            # Record dancer details
                                            split_msg = msg.split("|")
                                            dancer_id = split_msg[1]
                                            self.ultra96.init_dancer(dancer_id)
                                    elif ("[P]" in msg):
                                            if (self.ultra96.movements == ["-", "-", "-"]):
                                                self.movement_time = self.ultra96.begin_movement_time = int(float(time.time()))
                                            split_msg = msg.split("|")
                                            movement = split_msg[1]
                                            if (movement == "E"):
                                                self.isPredict = True
                                                time_stamp = split_msg[2]
                                                self.ultra96.logout_timestamps[dancer_id] = int(float(time_stamp)) - int(float(clock_offset))
                                                if (list(self.ultra96.logout_timestamps.values()).count(0) < 2):
                                                    self.ultra96.send_logout()
                                                    continue
                                                self.isPredict = False

                                            if (self.ultra96.movements[int(dancer_id) - 1] == "-"):
                                                #print(f"Dancer {dancer_id}: {movement}")
                                                self.ultra96.movements[int(dancer_id) - 1] = movement
                                    elif ("[O]" in msg):
                                            split_msg = msg.split("|")
                                            clock_offset = split_msg[1]
                                            to_print = f"[OFFSET] Offset for Dancer {dancer_id}: {clock_offset}"
                                            #print(to_print)
                                                
                                            


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
