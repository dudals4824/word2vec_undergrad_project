import numpy as np
import pandas as pd

def next_batch(start, batch_size, traindf, wv):
    # start : epoch에서의 index를 뜻함.
    batch_xs = list()
    batch_ys = list()

    for i in range(start, start+batch_size):
        arr = traindf.loc[i].values
        arr = arr[0].split(" ")
        x = []

        y = np.zeros(50095, dtype="int32")
        y[int(arr[11])] = 1
        batch_ys.append(y)

        for idx in range(0, 11):
            if idx == 5:
                continue
            x.append(wv.vectors[int(arr[idx])])
        batch_xs.append(x)


    start = start + batch_size

    return start, batch_xs, batch_ys