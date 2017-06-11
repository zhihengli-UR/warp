from torch.autograd.function import Function
import torch
from _ext import warp_lib


class WarpFunction(Function):
    def forward(self, input, flow):
        is_cuda = input.is_cuda or flow.is_cuda

        if is_cuda:
            input.cpu()
            flow.cpu()

        rec_image = torch.FloatTensor(input.size()).zero_()
        warp_lib.warpForward(input, flow, rec_image)

        self.save_for_backward(input, flow)

        if is_cuda:
            input.cuda()
            flow.cuda()

        return rec_image

    def backward(self, grad_output):
        assert not self.needs_input_grad[0]
        assert self.needs_input_grad[1]

        is_cuda = grad_output.is_cuda
        if is_cuda:
            grad_output.cpu()

        input, flow = self.saved_tensors

        if input.is_cuda:
            input.cpu()

        if flow.is_cuda:
            flow.cpu()

        # no need to calculate gradient for input
        grad_input = torch.FloatTensor(input.size()).zero_()
        grad_flow = torch.FloatTensor(flow.size()).zero_()

        warp_lib.warpBackward(input, flow, grad_flow, grad_output)

        if is_cuda:
            grad_input.cuda()
            grad_flow.cuda()

        return grad_input, grad_flow

    # def forward_python(self, input, flow):
    #     # two channels of optical flow
    #     assert flow.size()[1] == 2
    #     # same height and width
    #     assert flow.size()[2] == input.size()[2]
    #     assert flow.size()[3] == input.size()[3]
    #
    #     output = torch.FloatTensor(input.size()).zero_()
    #
    #     b, _, h, w = input.size()
    #     for i_b in range(b):
    #         for i in range(h):
    #             for j in range(w):
    #
    #                 i2 = i + flow[i_b][0][i][j]
    #                 j2 = j + flow[i_b][1][i][j]
    #
    #                 floor_i2 = int(math.floor(i2))
    #                 floor_j2 = int(math.floor(j2))
    #                 ceil_i2 = int(math.ceil(i2))
    #                 ceil_j2 = int(math.ceil(j2))
    #
    #                 if not ((floor_i2 >= 0) and (floor_j2 >= 0) and (ceil_i2 < h) and (ceil_j2 < w)):
    #                     for c in range(3):
    #                         output[i_b][c][i][j] = 0
    #                     continue
    #
    #                 theta_x = i2 - floor_i2
    #                 theta_y = j2 - floor_j2
    #                 for c in range(3):
    #                     value = (1 - theta_x) * (1 - theta_y) * input[i_b][c][floor_i2][floor_j2] + \
    #                                         theta_x * (1 - theta_y) * input[i_b][c][ceil_i2][floor_j2] + \
    #                                         (1 - theta_x) * theta_y * input[i_b][c][floor_i2][ceil_j2] + \
    #                                         theta_x * theta_y * input[i_b][c][ceil_i2][ceil_j2]
    #                     output[i_b][c][i][j] = int(math.floor(value))
    #
    #     return output


