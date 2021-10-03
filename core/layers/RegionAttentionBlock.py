import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class RAB(nn.Module):
    def __init__(self,
                 width,
                 channels,
                 class_num,
                 inner_dim,
                 hidden_channels=8):
        super(RAB, self).__init__()
        self.width = width
        self.class_num = class_num
        self.inner_dim = inner_dim
        self.classification = nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=class_num,kernel_size=5,stride=1,padding=2),
            nn.Softmax(1),
        )
        self.embedding_qk = nn.Sequential(
            Rearrange('b t c h w -> (b t) c h w'),
            nn.Conv2d(in_channels=channels,out_channels=hidden_channels,kernel_size=4,stride=4,padding=0),
            Rearrange('b c h w -> b (c h w)'),
            Rearrange('(b t) d -> b t d', t = self.class_num)
        )
        self.embedding_to_v = nn.Sequential(
            Rearrange('b t c h w -> (b t) c h w'),
            nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=5,stride=1,padding=2),
            Rearrange('b c h w -> b (c h w)'),
            Rearrange('(b t) d -> b t d', t=self.class_num)
        )
        self.to_q = nn.Linear(hidden_channels*64,inner_dim,bias=False)
        self.to_k = nn.Linear(hidden_channels*64,inner_dim,bias=False)
        self.out_embeding = Rearrange('b (c h w) -> b c h w',h = width,w=width)


    def forward(self, input):
        class_label = self.classification(input)
        classes = torch.unsqueeze(input,1)*torch.unsqueeze(class_label,2)
        x_qk = self.embedding_qk(classes)
        q = self.to_q(x_qk)
        k = self.to_k(x_qk)
        dot = torch.matmul(q,k.permute((0,2,1)))
        attn = torch.softmax(dot,1)
        v = self.embedding_to_v(classes)
        res = torch.matmul(attn,v)
        res = torch.sum(res,1)
        out = input+self.out_embeding(res)
        return out

