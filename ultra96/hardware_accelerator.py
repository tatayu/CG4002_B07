from pynq import Overlay, allocate
from config import *
import numpy as np

class HardwareAccelerator:
    def __init__(self):
        self.design = Overlay(DESIGN_PATH)
        self.design_dma = self.design.axi_dma_0
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
    
    # Allocates buffer for the input and output
    def buffer_allocate(self, inputs):
        # print(self.input_size)
        input_buffer = allocate(shape=(self.input_size,), dtype=np.float32)
        # print(input_buffer)
        for i in range(self.input_size):
            # print(i)
            input_buffer[i] = inputs[0][i]

        output_buffer = allocate(shape=(self.output_size,), dtype=np.float32)
        input_buffer.flush()
        return input_buffer, output_buffer
    
    # Transfers data from the buffer to a list
    def transfer_data(self, output_buffer):
        outputs =[]
        for i in range(output_buffer.size):
            outputs.append(output_buffer[i])
        return outputs
   
    def predict(self, test_input):
        # print(test_input)
        input_buffer, output_buffer = self.buffer_allocate(test_input)
        self.design_dma.sendchannel.transfer(input_buffer)
        self.design_dma.recvchannel.transfer(output_buffer)
        self.design_dma.sendchannel.wait()
        self.design_dma.recvchannel.wait()
        results = self.transfer_data(output_buffer)
        output_buffer.freebuffer()
        return results
