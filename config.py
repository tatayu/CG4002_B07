import os

directory = 'collected_data'
entries = sorted(os.listdir(directory))
entries_without_moves = []
entries_only_moves = []
moves = sorted(['stationary', 'move_left', 'move_right'])
for x in entries:
    flag = 1
    for y in moves:
        if y in x:
            flag = 0
    if flag:
        entries_without_moves.append(x)
    else:
        entries_only_moves.append(x)

entries = []
entries.extend(entries_without_moves)
entries.extend(entries_only_moves)
print(entries)

feature_list = [
    'mean', 
    'max', 
    'min', 
    'median', 
    'gradient', 
    'std', 
    'iqr', 
    # 'skew', 
    'zero_crossing',
    # 'cwt', 
    'no_peaks', 
    'recurring_dp', 
    # 'ratio_v_tsl', 
    # 'sum_recurring_dp', 
    'var_coeff', 
    'kurtosis'
]

testing_count = 20
window_size = 20
overlap = 18
clustering = False
deployed = False

dances = {}

for i,x in enumerate(entries):
    dances[i] = x[:-5]

print(dances)
# for k,v in dances.items():
#     print(f'No: {k+1}, Name: {v.rsplit("_")[0]}, Person: {v.rsplit("_")[-1]}',end='\t')
#     if k % 5 == 0 and k != 0:
#         print()