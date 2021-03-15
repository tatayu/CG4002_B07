import os

directory = 'collected_data'
entries = sorted(os.listdir(directory))

feature_list = [
    'mean', 
    'max', 
    'min', 
    'median', 
    # 'gradient', 
    'std', 
    # 'iqr', 
    # 'skew', 
    # 'zero_crossing',
    # 'cwt', 
    'no_peaks', 
    'recurring_dp', 
    # 'ratio_v_tsl', 
    # 'sum_recurring_dp', 
    'var_coeff', 
    'kurtosis'
]

testing_count = 20
window_size = 50

clustering = False
deployed = True

dances = {}

for i,x in enumerate(entries):
    dances[i] = x[:-5]

# for k,v in dances.items():
#     print(f'Dance move: {k+1}, Name: {v}')