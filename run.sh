# CG4002_B07
echo "disconnect 80:30:DC:E9:1C:2F" | bluetoothctl 
sleep 2
echo "disconnect 80:30:DC:D9:23:1E" | bluetoothctl 
sleep 2
echo "disconnect 34:B1:F7:D2:34:71" | bluetoothctl 
sleep 2
echo "exit" | bluetoothctl 
cd ~/CG4002_B07
python3 CommsInternal.py
