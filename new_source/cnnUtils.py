import numpy as np
import pandas as pd

def batch_iter(batch_size, num_epochs, traindf):
    data = traindf.values
    data_size = len(traindf)

    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        for batch_num in range(num_batches_per_epoch):
            x_zip = []
            y_zip = []
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            for i in range(start_index, end_index):
                input_x = [int(x) for x in data[i][0].split(" ")[0:10]]
                y = np.zeros(14920, dtype=np.int32)
                y[int(data[i][1])] = 1

                x_zip.append(input_x)
                y_zip.append(y)

            yield zip(x_zip, y_zip)

# def next_batch(start, batch_size, traindf):
#     # start : epoch에서의 index를 뜻함.
#     batch_xs = list()
#     batch_ys = list()
#
#     for i in range(start, start+batch_size):
#         df = traindf.loc[i].values
#         input_x = df[0].split(" ")
#         x = []
#
#         y = np.zeros(14920, dtype="int32")
#         y[int(df[1])] = 1
#         batch_ys.append(y)
#
#         batch_xs.append(x)
#
#     start = start + batch_size
#
#     return start, batch_xs, batch_ys