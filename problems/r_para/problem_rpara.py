from torch.utils.data import Dataset
import torch
import os
import pickle
import torch.nn as nn
from problems.r_para.state_rpara import StateRPARA
import numpy as np


# from utils.beam_search import beam_search


class RPARA(object):
    NAME = 'r_para'  #

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, ss_size, _ = dataset['ss_loc'].size()
        _, r_size, _ = dataset['return_loc'].size()

        # Gather dataset in order of tour
        # r_padding = nn.ZeroPad2d(padding=(0, 2, 0, 0))
        # robot_loc = r_padding(dataset['robot'])[:, None, :]
        robot_ss_return = torch.cat((torch.cat((dataset['robot'], dataset['ss_loc']), 1),
                                     dataset['return_loc']), 1)

        d = robot_ss_return.gather(1, pi[..., None].expand(*pi.size(), robot_ss_return.size(-1)))

        # 将变成回归位置的货架对应拣选站坐标清空
        for jj in range(d.size(1)):
            if jj % 2 == 1:
                d[:, jj, 2:] = d[:, jj, :2]
        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        distance = 0
        for i in range(d.size(1) - 1):
            dist = torch.abs(d[:, i, 0] - d[:, i, 2]) + torch.abs(d[:, i, 1] - d[:, i, 3]) + \
                   torch.abs(d[:, i, 2] - d[:, i + 1, 0]) + torch.abs(d[:, i, 3] - d[:, i + 1, 1])
            # dist = (d[:, i, :2] - d[:, i, 2:]).norm(p=2, dim=1) + (d[:, i, 2:] - d[:, i + 1, :2]).norm(p=2, dim=1)
            distance += dist

        # Depot to first and Last to depot, will be 0 if depot is last
        distance += torch.abs(d[:, 0, 0] - dataset['robot'][:, 0, 0]) + torch.abs(d[:, 0, 1] - dataset['robot'][:, 0, 1]) + \
                    torch.abs(d[:, -1, 2] - dataset['robot'][:, 0, 0]) + torch.abs(d[:, -1, 3] - dataset['robot'][:, 0, 1])

        return distance, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return R_PARADataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateRPARA.initialize(*args, **kwargs)

    # @staticmethod
    # def beam_search(input, beam_size, expand_size=None,
    #                 compress_mask=False, model=None, max_calc_batch_size=4096):
    #
    #     assert model is not None, "Provide model"
    #
    #     fixed = model.precompute_fixed(input)
    #
    #     def propose_expansions(beam):
    #         return model.propose_expansions(
    #             beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
    #         )
    #
    #     state = TSP.make_state(
    #         input, visited_dtype=torch.int64 if compress_mask else torch.uint8
    #     )
    #
    #     return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    robot, ss_loc, return_loc, *args = args
    grid_size = 1  # 1 x 1 的方格
    # if len(args) > 0:
    #     depot_types, customer_types, grid_size = args
    return {
        'robot': torch.tensor(robot, dtype=torch.float) / grid_size,
        'ss_loc': torch.tensor(ss_loc, dtype=torch.float) / grid_size,
        'return_loc': torch.tensor(return_loc, dtype=torch.float) / grid_size
    }


class R_PARADataset(Dataset):

    def __init__(self, filename=None, ss_size=6, r_size=6, num_samples=1000000, offset=0, distribution=None):
        super(R_PARADataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset + num_samples]]

        else:
            
            valid_row = np.array([2, 3, 6, 7, 10, 11, 14, 15])
            valid_col = np.array([2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27])
            stations = np.array([[6, 0], [11, 0], [17, 4], [17, 11], [17, 18], [17, 25], [11, 29], [6, 29]])

            self.data = []
            for i in range(num_samples):
                locs = []
                while len(locs) < (ss_size+r_size):
                    row = np.random.choice(valid_row)
                    col = np.random.choice(valid_col)
                    if [row, col] not in locs:
                        locs.append([row, col])
                robot = torch.FloatTensor([np.random.randint(0, 18), np.random.randint(0, 29)])
                # robot = torch.zeros(2)
                shelf_loc = torch.FloatTensor(locs[:ss_size])
                return_loc = torch.FloatTensor(locs[ss_size:])

                station_index = np.random.choice(np.arange(len(stations)), ss_size, replace=True)
                station_loc = torch.FloatTensor(stations[station_index])

                ss_loc = torch.cat((shelf_loc, station_loc), -1)

                self.data.append({
                    'robot': robot,
                    'ss_loc': ss_loc,
                    'return_loc': return_loc
                })
            

            # self.data = [
            #     {
            #         'robot': torch.FloatTensor(2).random_(0, 100),  # 可能生出相同的数
            #         'ss_loc': (torch.FloatTensor(ss_size, 4).random_(0, 100)),
            #         'return_loc': torch.FloatTensor(r_size, 2).random_(0, 100)
            #     }
            #     for i in range(num_samples)
            # ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
