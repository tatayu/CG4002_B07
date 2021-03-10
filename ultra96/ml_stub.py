import threading
import time
import random

class MLStub(threading.Thread):
	def __init__(self, ultra96):
		super(MLStub, self).__init__()

		self.ultra96 = ultra96
		
	def output_move(self, data):
		print(type(data))
		position = [1, 2, 3]
		actions = ["gun", "sidepump", "hair"]
		sync = [1.23, 2.13, 3.12]

		for x in range(len(actions)):
			random.shuffle(actions)
			random.shuffle(position)
			random.shuffle(sync)

		return position[0], position[1], position[2], actions[0], sync[0]


if __name__ == '__main__':
	pass