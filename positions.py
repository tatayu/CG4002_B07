import numpy as np
import json
import pandas as pd
from scipy import stats
'''
8: Move Left
9: Move Right
'''
def moves_after_round(p,r):
    m = None
    if stats.mode(r).count[0] > 1 and stats.mode(r).mode[0] < 8:
        m = stats.mode(r).mode[0]
    elif r[0] == 9 and r[2] == 8:
        p = [p[2],p[1],p[0]]    
    elif r[1] == 9 and r[2] == 8:
        p = [p[0],p[2],p[1]]        
    elif r[0] == 9 and r[1] == 8:
        p = [p[1],p[0],p[2]]   
    else:
        print("Error in Data")   
        return p,m          
    print("Position:", p, "Dance:", m)
    return p,m
def moves_template():
    return np.array(
        [
            [9,8,7], # 2,1,3
            [1,1,2], # Dance 1
            [2,2,3], # Dance 2
            [3,3,3], # Dance 3
            [4,4,4], # Dance 4
            [7,9,8], # 2,3,1
            [5,1,2], # Error in Data
            [6,2,3], # Error in Data
            [7,3,3], # Dance 3
            [8,4,4], # Dance 4
            [6,6,3], # Dance 6
            [8,9,7], # Error in Data
            [8,9,8], # 2,1,3
            [9,9,7]  # Error in Data
        ]
    )

def main():
    arr = moves_template()
    pos = np.array([1,2,3])
    for r in arr:
        pos, move = moves_after_round(pos, r)
    
    f = open('collected_data/sidepump_john.json',) 
    test_json = json.load(f)
    
    test_df = pd.DataFrame.from_dict(test_json)
    print(test_df.shape)
main()