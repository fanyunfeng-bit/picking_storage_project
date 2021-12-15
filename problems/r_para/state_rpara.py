import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
import numpy as np


class StateRPARA(NamedTuple):
    # Fixed input
    # coords: torch.Tensor  # robot + shelf_loc + station_loc + return_loc
    robot_loc: torch.Tensor  # 单机---类比为depot
    ss_loc: torch.Tensor  # shelf-station 二元组 n x 1 x 4，分别为二者的坐标
    return_loc: torch.Tensor  # m x 1 x 2
    n_loc: torch.Tensor  # n + m

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    # first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited

    visited_shelf: torch.Tensor  # record the shelves that have been visited
    visited_return: torch.Tensor  # record the return locs that have been visited(shelves have been visited would be return locs)

    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.ss_loc.size(-2))

    # @property
    # def dist(self):
    #     return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ss_loc=self.ss_loc[key],
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                # used_capacity=self.used_capacity[key],
                visited_=self.visited_[key],
                visited_shelf=self.visited_shelf[key],
                visited_return=self.visited_return[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
            )
        return super(StateRPARA, self).__getitem__(key)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        robot = input['robot']
        ss_loc = input['ss_loc']
        return_loc = input['return_loc']

        batch_size, n_shelf, _ = ss_loc.size()
        _, n_return, _ = return_loc.size()
        n_loc = n_shelf + n_return

        return StateRPARA(
            # coords=torch.cat((torch.cat((torch.cat((robot[:, None, :], shelf_loc), -2), station_loc), -2), return_loc), -2),
            robot_loc=robot[:, None, :],
            ss_loc=ss_loc,
            # station_loc=station_loc,
            return_loc=return_loc,
            n_loc=n_loc,
            ids=torch.arange(batch_size, dtype=torch.int64, device=ss_loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=ss_loc.device),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,  # +1 加上robot的起始位置
                    dtype=torch.uint8, device=ss_loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=ss_loc.device)  # Ceil
            ),
            visited_shelf=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc + 1,  # +1 加上robot的起始位置
                    dtype=torch.uint8, device=ss_loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=ss_loc.device)  # Ceil
            ),
            visited_return=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,  # +1 加上robot的起始位置
                    dtype=torch.uint8, device=ss_loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=ss_loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=ss_loc.device),
            cur_coord=input['robot'][:, None, :],  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=ss_loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()

        # Manhattan距离
        return self.lengths + torch.abs(self.robot_loc[self.ids, 0, 0] - self.cur_coord[0]) + \
               torch.abs(self.robot_loc[self.ids, 0, 1] - self.cur_coord[1])  # 加上回到初始位置的距离

    def all_finished(self):
        # print(self.i.item() >= self.demand.size(-1) and self.visited.all())
        # 选择一个shelf对应一个return位，shelf与station的关系确定，self.i只在选择shelf和return时加一！！！！
        return self.i.item() >= self.ss_loc.size(-2) * 2 and self.visited_[:, :, 1:self.ss_loc.size(1) + 1].all()

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected
        n_shelf = self.ss_loc.size(-2)  #
        n_return = self.return_loc.size(-2)

        # Add the length
        locs = torch.cat((self.ss_loc, self.return_loc), 1)
        cur_coord = locs[self.ids, selected-1]

        if cur_coord.size(-1) == 4:
            lengths = self.lengths + torch.abs(cur_coord[:, :, 0] - self.cur_coord[0]) + \
                      torch.abs(cur_coord[:, :, 1] - self.cur_coord[1]) + \
                      torch.abs(cur_coord[:, :, 0] - cur_coord[:, :, 2]) + torch.abs(
                cur_coord[:, :, 1] - cur_coord[:, :, 3])
            c_c = cur_coord[:, :, 2:]  # 将station的位置作为robot的当前位置
        else:
            lengths = self.lengths + torch.abs(cur_coord[0] - self.cur_coord[0]) + \
                      torch.abs(cur_coord[1] - self.cur_coord[1])
            c_c = cur_coord

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
            if self.i % 2 == 1:
                visited_return = self.visited_return.scatter(-1, prev_a[:, :, None], 1)
                visited_shelf = self.visited_shelf
            else:
                visited_shelf = self.visited_shelf.scatter(-1, prev_a[:, :, None], 1)
                visited_return = self.visited_return
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)
            if self.i % 2 == 1:
                visited_return = mask_long_scatter(self.visited_return, prev_a - 1)
                visited_shelf = self.visited_shelf
            else:
                visited_shelf = mask_long_scatter(self.visited_shelf, prev_a - 1)
                visited_return = self.visited_return

        if (selected <= n_shelf).all():
            self.ss_loc[self.ids, selected - 1, 2:] = self.ss_loc[self.ids, selected - 1, :2]  # 被选择过的shelf对应station清空
            # print('self.ss_loc: ', self.ss_loc)
            # print(self.ids)

        return self._replace(
            prev_a=prev_a, visited_=visited_, ss_loc=self.ss_loc, visited_shelf=visited_shelf,
            visited_return=visited_return, lengths=lengths, cur_coord=c_c, i=self.i + 1
        )

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_
            visited_return = self.visited_return
            visited_shelf = self.visited_shelf
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.n_loc)
            visited_return = mask_long2bool(self.visited_return, n=self.n_loc)
            visited_shelf = mask_long2bool(self.visited_shelf, n=self.n_loc)

            # Nodes that cannot be visited are already visited or shouldn't be served now
        batch_size, n_shelf, _ = self.ss_loc.size()
        mask_serve = torch.zeros(
            batch_size, 1, np.array(self.n_loc) + 1,  # +1 加上robot的起始位置
            dtype=torch.uint8, device=self.ss_loc.device
        )
        if self.i % 2 == 0:
            mask_serve[:, :, n_shelf + 1:] = 1
            mask_loc = visited_loc | mask_serve
        else:
            mask_serve[:, :, 1:n_shelf + 1] = 1
            mask_serve[:, :, 1:n_shelf + 1] = visited_shelf[:, :, 1:n_shelf + 1] ^ mask_serve[:, :, 1:n_shelf + 1]
            mask_loc = mask_serve | visited_return

        # Cannot visit the initial robot loc if the task are not completed totally.
        mask_robot = torch.zeros(
            batch_size, 1, np.array(self.n_loc) + 1,  # +1 加上robot的起始位置
            dtype=torch.uint8, device=self.ss_loc.device
        )
        if not self.all_finished():
            mask_robot[:, :, 0] = 1
        mask_loc |= mask_robot

        return mask_loc, visited_return
