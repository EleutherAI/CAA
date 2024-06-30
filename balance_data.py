import json
import numpy as np

import os, sys
# dir2 = os.path.abspath('')
# dir1 = dir  # os.path.dirname(dir2)
# if not dir1 in sys.path: sys.path.append(dir1)

from behaviors import (
    get_open_response_data_path,
    ALL_BEHAVIORS,
)

def rebalance(data):
    n = len(data)

    # get exact 50-50 split, not IID
    n1 = n // 2
    n2 = n - n1

    # get indices
    idx1 = np.random.choice(n, n1, replace=False)

    new_data = []

    #  refrigerator?\n\nChoices:\n (A) Yes, the `--
    
    for i in range(n):
        if i in idx1:
            answer_a = data[i]['answer_matching_behavior']
            answer_b = data[i]['answer_not_matching_behavior']
        else:
            answer_a = data[i]['answer_not_matching_behavior']
            answer_b = data[i]['answer_matching_behavior']
        new_data.append({
            "question": f"{data[i]['question']}\n\nChoices:\n (A) {answer_a}\n (B) {answer_b}",
            "answer_matching_behavior": ("(A)" if i in idx1 else "(B)"),
            "answer_not_matching_behavior": ("(B)" if i in idx1 else "(A)"),
        })

    return new_data


for behavior in ALL_BEHAVIORS:

    path = get_open_response_data_path(behavior)

    # json
    with open(path, 'r') as f:
        data = json.load(f)

    new_data = rebalance(data)
    new_path = path.replace('openresponse', 'balanced')

    with open(new_path, 'w') as f:
        assert 'balanced' in new_path
        print(f'Writing to {new_path}')
        json.dump(new_data, f, indent=2)
