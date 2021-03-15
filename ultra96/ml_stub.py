import threading
import time
import random

class MLStub(threading.Thread):
	def __init__(self):
		super(MLStub, self).__init__()
		
	def predict(self, data):
		position = [1, 2, 3]
		actions = ["gun", "sidepump", "hair"]
		sync = [1.23, 2.13, 3.12]

		for x in range(len(actions)):
			random.shuffle(actions)
			random.shuffle(position)
			random.shuffle(sync)

		return actions[0]


if __name__ == '__main__':
	pass