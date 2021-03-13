# CG4002 External Comms

## Laptop
- laptop_main.py
  - Main driver for laptops, calls laptop_client and int_comms
- laptop_client.py
  - Handles TCP connection and message transmission between laptops and ultra96
- int_comms_stub.py
  - Stub for int_comms integration
- test.csv
  - Test data (300 samples, input size=30)
- config.py
  - Contains values and addresses for sshtunnel, etc
- requirements.txt

## Ultra96
- ultra96_main.py
  - Main driver for ultra96, calls ultra96_client and ultra96_server as well as HardwareAccelerator. Handles information sharing between the threads.
- ultra96_client.py
  - Handles communication between ultra96 and evaluation server
- ultra96_server.py
  - Handles communication between laptops and ultra96 (num_laptops variable determines number of dancers to connect to the server)
- hardware_accelerator.py
  - ML Class, takes in input data and outputs dance move
- ml_stub.py
  - Stub for ML integration
- dummy_eval_server.py
  - Altered evaluation server script for testing purposes
- config.py
  - Contains values and addresses for TCP, HardwareAccelerator etc
- requirements.txt
