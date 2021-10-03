import torch
import torch.nn as nn
from core.layers.RAPCell_h import RAPCellh
from core.layers.RAPCell_x import RAPCellx
from core.layers.RAPCell import RAPCell
from core.layers.STLSTMCell import ST_LSTM_Cell

class PredRNN(nn.Module):
    def __init__(self, configs):
        super(PredRNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        num_layers = len(num_hidden)
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            cell_list.append(
                ST_LSTM_Cell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, frames, mask_true):
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).cuda()
        for t in range(self.configs.total_length-1):

            if t < self.configs.input_length:
                net = frames[:,t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers-1])
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=1)

        return next_frames

class RAP_Net(nn.Module):
    def __init__(self, configs):
        super(RAP_Net, self).__init__()
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        num_layers = len(num_hidden)
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size
        patch_sizes = [4,4,4,4]
        layer_norm_list = []
        for i in range(num_layers):
            layer_norm_list.append(
                nn.LayerNorm([num_hidden[i],width,width])
            )
        self.layer_norm_list = nn.ModuleList(layer_norm_list)
        cnns_list = []
        for i in range(num_layers):

            in_channel = (self.configs.total_length-1)*(configs.patch_size**2) if i == 0 else num_hidden[i-1]

            cnns_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channel,num_hidden[i],configs.filter_size,1,configs.filter_size // 2),
                    nn.LayerNorm([num_hidden[i], width, width]),
                    nn.ReLU()
                )
            )

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            cell_list.append(
                RAPCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, patch_sizes[i], configs.layer_norm)
            )

        # print()
        self.cnns_list = nn.ModuleList(cnns_list)
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)



    def attn_channel(self,in_query,in_keys,in_values):
        q_shape = in_query.shape
        k_shape = in_keys.shape
        batch = q_shape[0]
        num_channels = q_shape[1]
        width = q_shape[2]
        height = q_shape[3]
        length = k_shape[1]
        query = in_query.reshape([batch,num_channels,-1])
        key = in_keys.reshape([batch,-1,height*width]).permute((0, 2, 1))
        value = in_values.reshape([batch,-1,height*width]).permute((0, 2, 1))

        attn = torch.matmul(query,key)
        attn = torch.nn.Softmax(dim=-1)(attn)
        attn = torch.matmul(attn,value.permute(0,2,1))
        attn = attn.reshape([batch,num_channels,width,height])

        return attn

    def forward(self, frames, mask_true):
        B,T,C,H,W = frames[:,1:].squeeze(2).shape
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).cuda()
        empty_cnn_inputs = torch.zeros((B,T*C,H,W)).cuda()
        for t in range(self.configs.total_length-1):

            if t < self.configs.input_length:
                net = frames[:,t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            empty_cnn_inputs[:,t*C:(t+1)*C] = net
            cnn_input = empty_cnn_inputs.clone()

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)
            cnn_output = self.cnns_list[0](cnn_input)
            h_t[0] =  h_t[0] + self.attn_channel(h_t[0],cnn_output,cnn_output)
            cnn_input = cnn_output
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                cnn_output = self.cnns_list[i](cnn_input)
                h_t[i] =  h_t[i] + self.attn_channel(h_t[i], cnn_output, cnn_output)
                cnn_input = cnn_output

            x_gen = self.conv_last(h_t[self.num_layers-1])
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=1)

        return next_frames


class RAP_Cell(nn.Module):
    def __init__(self, configs):
        super(RAP_Cell, self).__init__()
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        num_layers = len(num_hidden)
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size
        patch_sizes = [4, 4, 4, 4]
        layer_norm_list = []
        for i in range(num_layers):
            layer_norm_list.append(
                nn.LayerNorm([num_hidden[i], width, width])
            )
        self.layer_norm_list = nn.ModuleList(layer_norm_list)

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                RAPCell(in_channel, num_hidden[i], width, configs.filter_size,
                              configs.stride, patch_sizes[i], configs.layer_norm)
            )

        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def attn_channel(self, in_query, in_keys, in_values):
        q_shape = in_query.shape
        k_shape = in_keys.shape
        batch = q_shape[0]
        num_channels = q_shape[1]
        width = q_shape[2]
        height = q_shape[3]
        length = k_shape[1]
        query = in_query.reshape([batch, num_channels, -1])
        key = in_keys.reshape([batch, -1, height * width]).permute((0, 2, 1))
        value = in_values.reshape([batch, -1, height * width]).permute((0, 2, 1))

        attn = torch.matmul(query, key)
        attn = torch.nn.Softmax(dim=-1)(attn)
        attn = torch.matmul(attn, value.permute(0, 2, 1))
        attn = attn.reshape([batch, num_channels, width, height])

        return attn

    def forward(self, frames, mask_true):



        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).cuda()

        for t in range(self.configs.total_length - 1):

            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=1)

        return next_frames


class RAP_Cell_h(nn.Module):
    def __init__(self, configs):
        super(RAP_Cell_h, self).__init__()
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        num_layers = len(num_hidden)
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size
        patch_sizes = [4, 4, 4, 4]
        layer_norm_list = []
        for i in range(num_layers):
            layer_norm_list.append(
                nn.LayerNorm([num_hidden[i], width, width])
            )
        self.layer_norm_list = nn.ModuleList(layer_norm_list)

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                RAPCellh(in_channel, num_hidden[i], width, configs.filter_size,
                              configs.stride, patch_sizes[i], configs.layer_norm)
            )

        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def attn_channel(self, in_query, in_keys, in_values):
        q_shape = in_query.shape
        k_shape = in_keys.shape
        batch = q_shape[0]
        num_channels = q_shape[1]
        width = q_shape[2]
        height = q_shape[3]
        length = k_shape[1]
        query = in_query.reshape([batch, num_channels, -1])
        key = in_keys.reshape([batch, -1, height * width]).permute((0, 2, 1))
        value = in_values.reshape([batch, -1, height * width]).permute((0, 2, 1))

        attn = torch.matmul(query, key)
        attn = torch.nn.Softmax(dim=-1)(attn)
        attn = torch.matmul(attn, value.permute(0, 2, 1))
        attn = attn.reshape([batch, num_channels, width, height])

        return attn

    def forward(self, frames, mask_true):



        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).cuda()

        for t in range(self.configs.total_length - 1):

            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=1)

        return next_frames


class RAP_Cell_x(nn.Module):
    def __init__(self, configs):
        super(RAP_Cell_x, self).__init__()
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        num_layers = len(num_hidden)
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size
        patch_sizes = [4, 4, 4, 4]
        layer_norm_list = []
        for i in range(num_layers):
            layer_norm_list.append(
                nn.LayerNorm([num_hidden[i], width, width])
            )
        self.layer_norm_list = nn.ModuleList(layer_norm_list)

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                RAPCellx(in_channel, num_hidden[i], width, configs.filter_size,
                              configs.stride, patch_sizes[i], configs.layer_norm)
            )

        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def attn_channel(self, in_query, in_keys, in_values):
        q_shape = in_query.shape
        k_shape = in_keys.shape
        batch = q_shape[0]
        num_channels = q_shape[1]
        width = q_shape[2]
        height = q_shape[3]
        length = k_shape[1]
        query = in_query.reshape([batch, num_channels, -1])
        key = in_keys.reshape([batch, -1, height * width]).permute((0, 2, 1))
        value = in_values.reshape([batch, -1, height * width]).permute((0, 2, 1))

        attn = torch.matmul(query, key)
        attn = torch.nn.Softmax(dim=-1)(attn)
        attn = torch.matmul(attn, value.permute(0, 2, 1))
        attn = attn.reshape([batch, num_channels, width, height])

        return attn

    def forward(self, frames, mask_true):



        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).cuda()

        for t in range(self.configs.total_length - 1):

            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=1)

        return next_frames