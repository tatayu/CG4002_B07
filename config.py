import os

directory = 'Data-sets/Dance_Data'
entries = os.listdir(directory)

feature_list = ['mean', 'max', 'min', 'median', 'gradient', 'std', 
'skew', 'iqr', 'zero_crossing_counts', 'dominant_frequency']

testing_count = 20
window_size = 100

clustering = False
deployed = False
testing = False

dances = {}

for i,x in enumerate(entries):
    dances[i] = x[:-4]

for k,v in dances.items():
    print(f'Dance move: {k}, Name: {v}')