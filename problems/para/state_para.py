import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
import numpy as np


class StatePARA(NamedTuple):
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
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                # used_capacity=self.used_capacity[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
            )
        return super(StatePARA, self).__getitem__(key)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        robot = input['robot']
        ss_loc = input['ss_loc']
        # station_loc = input['station_loc']
        return_loc = input['return_loc'][:, :, :2]

        batch_size, n_shelf, _ = ss_loc.size()
        # _, n_station, _ = station_loc.size()
        _, n_return, _ = return_loc.size()
        n_loc = n_shelf + n_return

        return StatePARA(
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
        # print('selected: ', selected)
        # Update the state
        selected = selected[:, None]  # Add dimension for step
        # print(selected)
        prev_a = selected
        n_shelf = self.ss_loc.size(-2)  #
        # n_station = self.station_loc.size(-2)
        n_return = self.return_loc.size(-2)

        # Add the length
        if selected[0] <= n_shelf:
            cur_coord = self.ss_loc[self.ids, selected - 1]  # 当前的选择位置：shelf
        else:
            cur_coord = self.return_loc[self.ids, selected - n_shelf - 1]  # 或return位
        # print(cur_coord)
        # cur_coord =
        # cur_coord = self.coords.gather(
        #     1,
        #     selected[:, None].expand(selected.size(0), 1, self.coords.size(-1))
        # )[:, 0, :]

        # 此处self.cur_coord是上一个位置：与选择的shelf对应的station或者return位
        if cur_coord.size(-1) == 4:
            lengths = self.lengths + torch.abs(cur_coord[:, :, 0] - self.cur_coord[0]) + \
                      torch.abs(cur_coord[:, :, 1] - self.cur_coord[1]) + \
                      torch.abs(cur_coord[:, :, 0] - cur_coord[:, :, 2]) + torch.abs(
                cur_coord[:, :, 1] - cur_coord[:, :, 3])
            # lengths = self.lengths + (cur_coord[:, :, 0:2] - self.cur_coord).norm(p=2, dim=-1) + \
            #           (cur_coord[:, :, 0:2] - cur_coord[:, :, 2:]).norm(p=2, dim=-1)
            c_c = cur_coord[:, :, 2:]  # 将station的位置作为robot的当前位置
        else:
            lengths = self.lengths + torch.abs(cur_coord[0] - self.cur_coord[0]) + \
                torch.abs(cur_coord[1] - self.cur_coord[1])
            # lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)
            c_c = cur_coord

        # if selected == 0:
        #     lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)
        #     c_c = cur_coord
        # elif selected <= n_shelf:  # ->shelf->station
        #     lengths = self.lengths + (cur_coord[0:2] - self.cur_coord).norm(p=2, dim=-1) + \
        #               (cur_coord[0:2] - cur_coord[2:]).norm(p=2, dim=-1)
        #     c_c = cur_coord[3:]
        # else:
        #     lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)
        #     c_c = cur_coord[:2]

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        # selected_demand = self.demand.gather(-1, torch.clamp(prev_a - 1, 0, n_loc - 1))
        # selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)]

        # Increase capacity if depot is not visited, otherwise set to 0
        # used_capacity = torch.where(selected == 0, 0, self.used_capacity + selected_demand)
        # used_capacity = (self.used_capacity + selected_demand) * (prev_a != 0).float()

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        return self._replace(
            prev_a=prev_a, visited_=visited_,
            lengths=lengths, cur_coord=c_c, i=self.i + 1
        )

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.n_loc)

        # Nodes that cannot be visited are already visited or shouldn't be served now
        batch_size, n_shelf, _ = self.ss_loc.size()
        mask_serve = torch.zeros(
            batch_size, 1, np.array(self.n_loc) + 1,  # +1 加上robot的起始位置
            dtype=torch.uint8, device=self.ss_loc.device
        )
        if self.i % 2 == 0:
            mask_serve[:, :, n_shelf + 1:] = 1
        else:
            mask_serve[:, :, 1:n_shelf + 1] = 1
        # print('server: ', mask_serve, mask_serve.shape)
        # print('visited: ', visited_loc, visited_loc.shape)
        mask_loc = visited_loc | mask_serve

        # Cannot visit the initial robot loc if the task are not completed totally.
        mask_robot = torch.zeros(
            batch_size, 1, np.array(self.n_loc) + 1,  # +1 加上robot的起始位置
            dtype=torch.uint8, device=self.ss_loc.device
        )
        if not self.all_finished():
            mask_robot[:, :, 0] = 1
        mask_loc |= mask_robot
        # print(mask_loc, mask_loc.shape)
        return mask_loc, visited_loc
