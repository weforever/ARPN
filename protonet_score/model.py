import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd

data_u = pd.read_csv(r'D:/protonet_1/data_zhinengche.csv')
data_use = data_u.loc[:,
           ['车速', '累计里程', '总电压', '总电流', 'SOC', 'DC-DC状态', '挡位', '挡位驱动力', '挡位制动力',
            '最高电压电池单体代号', '电池单体电压最高值', '最低电压电池单体代号', '电池单体电压最低值',
            '最高温度探针单体代号', '最高温度值', '最低温度探针子系统代号', '最低温度值',
            '可充电储能装置故障总数']]
X = data_use.copy()
y = X.pop('可充电储能装置故障总数')

input_dim = X.shape[1]
# 定义Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 2)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(2, 6),
            nn.ReLU(),
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, 20),
            nn.ReLU(),
            nn.Linear(20, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 定义Prototypical Network
autoencoder = Autoencoder()

