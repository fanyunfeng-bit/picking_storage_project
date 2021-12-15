import numpy as np
import torch

valid_row = np.array([2, 3, 6, 7, 10, 11, 14, 15])
valid_col = np.array([2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27])
stations = np.array([[6, 0], [11, 0], [17, 4], [17, 11], [17, 18], [17, 25], [11, 29], [6, 29]])
num_samples = 2
ss_size = 5
r_size = 5

data = []
for i in range(num_samples):
    locs = []
    while len(locs) <= (ss_size+r_size):
        row = np.random.choice(valid_row)
        col = np.random.choice(valid_col)
        if [row, col] not in locs:
            locs.append([row, col])
    robot = torch.zeros(2)
    shelf_loc = torch.FloatTensor(locs[:ss_size])
    return_loc = torch.FloatTensor(locs[ss_size:])

    station_index = np.random.choice(np.arange(len(stations)), ss_size, replace=True)
    station_loc = torch.FloatTensor(stations[station_index])

    ss_loc = torch.cat((shelf_loc, station_loc), -1)

    data.append({
        'robot': robot,
        'ss_loc': ss_loc,
        'return_loc': return_loc
    })

print(data)

# data = [
#     {
#         'robot': torch.FloatTensor(2).random_(0, 100),  # 可能生出相同的数
#         'ss_loc': (torch.FloatTensor(ss_size, 4).random_(0, 100)),
#         'return_loc': torch.FloatTensor(r_size, 2).random_(0, 100)
#     }
#     for i in range(2)
# ]
# print(data)

aa = [1, 13,  5,  5,  8, 11,  4, 20,  9,  1, 10, 10,  7,  4,  6,  6,  2, 16, 3,  2]