import threading
import time

class IntCommsStub(threading.Thread):
	def __init__(self, laptop):
		super(IntCommsStub, self).__init__()

		self.laptop = laptop

	def run(self):
		while(True):
			data = "123 43 903724 23424"

			self.laptop.data_queue.put(data)
			time.sleep(10)

if __name__ == '__main__':
	pass