import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Generate import Generate, Generate_test, Generate_test_pos
from model import Autoencoder
from data import load_data
import torch.nn.functional as F
import torch.optim as optim
import random

from sklearn.model_selection import train_test_split

# data_u = pd.read_csv(r'D:/protonet_1/data_zhinengche.csv')
# data_use = data_u.loc[:,
#            ['车速', '累计里程', '总电压', '总电流', 'SOC', 'DC-DC状态', '挡位', '挡位驱动力', '挡位制动力',
#             '最高电压电池单体代号', '电池单体电压最高值', '最低电压电池单体代号', '电池单体电压最低值',
#             '最高温度探针单体代号', '最高温度值', '最低温度探针子系统代号', '最低温度值',
#             '可充电储能装置故障总数']]
# X = data_use.copy()
# y = X.pop('可充电储能装置故障总数')
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
#
# X_train = np.asarray(X_train)
# # q:X_train的维数是(248,10)
# X_test = np.asarray(X_test)
# # print(X_train)
# print(X_test)
# y_train = np.asarray(y_train)
# q:y_train的维数是(248,1)
# y_test = np.asarray(y_test)
# print(y_train)
# print(y_train.sum())
# print(y_test)
# print(y_test.sum())

# 使用StandardScaler对二维数组进行标准化处理
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# q:X_train_scaled的维数是(248,10)
# X_test_scaled = scaler.fit_transform(X_test)
# print(X_train_scaled)
X_train_scaled, y_train, X_test_scaled, y_test = load_data()
# 取248条故障数据放在前面train
sort_index = np.argsort(-y_train)
# print('y_train,t_test')
# print(y_train.sum())
# print(y_test.sum())
X_sorted = X_train_scaled[sort_index]
# print('X_sorted_train.shape')
# print(X_sorted.shape)
y_sorted = y_train[sort_index]

# 取126条故障数据放在前面test
sort_index_test = np.argsort(-y_test)
X_sorted_test = X_test_scaled[sort_index_test]
# print('X_sorted_test.shape')
# print(X_sorted_test.shape)
y_sorted_test = y_test[sort_index_test]


features = X_sorted

global last_prototype_vectors
global query_embeddings_labels

# def Generate(features):
#     N = N_num  # 选出500条数据，248条故障数据，252条正常数据，求两个原型向量
#     N_pos = pos_num  # 248条故障数据
#     support_input = torch.zeros(size=[support_num, features.shape[1]])  # 支持集
#     query_input = torch.zeros(size=[query_num, features.shape[1]])  # 查询集
#     postive_list = list(range(0, N_pos))  # 少
#     negtive_list = list(range(N_pos, N))  # 多
#     support_list_pos = random.sample(postive_list, support_pos_sample)  # 从248条故障数据中随机选出64条,作为支持集正例
#     support_list_neg = random.sample(negtive_list, support_neg_sample)  # 从252条正常数据中随机选出64条，作为支持集负例
#     query_list_pos = [item for item in postive_list if item not in support_list_pos]  # 184，作为查询集正例（248-64）
#     query_list_neg = [item for item in negtive_list if item not in support_list_neg]  # 188，作为查询集负例（252-64）
#     index = 0
#     for item in support_list_pos:
#         support_input[index, :] = features[item, :]
#         index += 1
#     for item in support_list_neg:
#         support_input[index, :] = features[item, :]
#         index += 1
#     index = 0
#     for item in query_list_pos:
#         query_input[index, :] = features[item, :]
#         index += 1
#     for item in query_list_neg:
#         query_input[index, :] = features[item, :]
#         index += 1
#     ones = torch.ones(pos_num - support_pos_sample, dtype=torch.long)
#     zeros = torch.zeros(neg_num - support_neg_sample, dtype=torch.long)
#     result = torch.cat((ones, zeros)).float().requires_grad_(True)
#     query_label = result
#     return support_input, query_input, query_label
#
#
# def Generate_test(features):
#     N = N_num_test  # 选出500条数据，248条故障数据，252条正常数据，求两个原型向量
#     N_pos = pos_num_test  # 248条故障数据
#     support_input_test = torch.zeros(size=[support_num_test, features.shape[1]])  # 支持集
#     query_input_test = torch.zeros(size=[query_num_test, features.shape[1]])  # 查询集
#     postive_list = list(range(0, N_pos))  # 少
#     negtive_list = list(range(N_pos, N))  # 多
#     support_list_pos = random.sample(postive_list, support_pos_sample)  # 从248条故障数据中随机选出64条,作为支持集正例
#     support_list_neg = random.sample(negtive_list, support_neg_sample)  # 从252条正常数据中随机选出64条，作为支持集负例
#     query_list_pos = [item for item in postive_list if item not in support_list_pos]  # 184，作为查询集正例（248-64）
#     query_list_neg = [item for item in negtive_list if item not in support_list_neg]  # 188，作为查询集负例（252-64）
#     index = 0
#     for item in support_list_pos:
#         support_input_test[index, :] = features[item, :]
#         index += 1
#     for item in support_list_neg:
#         support_input_test[index, :] = features[item, :]
#         index += 1
#     index = 0
#     for item in query_list_pos:
#         query_input_test[index, :] = features[item, :]
#         index += 1
#     for item in query_list_neg:
#         query_input_test[index, :] = features[item, :]
#         index += 1
#     ones = torch.ones(pos_num_test - support_pos_sample, dtype=torch.long)
#     zeros = torch.zeros(neg_num_test - support_neg_sample, dtype=torch.long)
#     result = torch.cat((ones, zeros)).float().requires_grad_(True)
#     query_label_test = result
#     return support_input_test, query_input_test, query_label_test

test_pos = 126
# def Generate_test_pos(features):
#     N = test_pos  # 选出500条数据，248条故障数据，252条正常数据，求两个原型向量
#     test_pos_input = torch.zeros(size=[test_pos, features.shape[1]])  # 获取全部的正例
#     test_pos_list = list(range(0, N))  # 126条正例
#     index = 0
#     for item in test_pos_list:
#         test_pos_input[index, :] = features[item, :]
#         index += 1
#     ones = torch.ones(test_pos, dtype=torch.long)
#     test_query_label = ones
#     return test_pos_input, test_query_label

N_num = 4000
pos_num = 248
neg_num = N_num - pos_num  # 多
support_pos_sample = 64  # 64
support_neg_sample = 64  # 64
support_num = support_pos_sample + support_neg_sample
query_num = N_num - support_num
query_pos = pos_num - support_pos_sample
query_neg = neg_num - support_neg_sample

X_sorted = torch.from_numpy(X_sorted)  # 将train数组转换成张量
# q:X_sorted的维数是多少
# q:X_sorted的维数是(248,10)
X_sorted_test = torch.from_numpy(X_sorted_test)  # 将test数组转换成张量
y_sorted = torch.from_numpy(y_sorted)  # 将train数组转换成张量
y_sorted_test = torch.from_numpy(y_sorted_test)  # 将test数组转换成张量

# 获取train支持集，查询集，查询集标签
support_input, query_input, query_label = Generate(X_sorted)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 转换为numpy数组，便于绘图
query_input_np = query_input.numpy()
query_label_np = query_label.detach().numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 用不同的标记绘制标签0和1的数据点
# ax.scatter(query_input_np[query_label_np == 0, 2], query_input_np[query_label_np == 0, 3], query_input_np[query_label_np == 0, 14], marker='+', label='Label 0')
# ax.scatter(query_input_np[query_label_np == 1, 2], query_input_np[query_label_np == 1, 3], query_input_np[query_label_np == 1, 14], marker='x', label='Label 1')
#
# ax.set_xlabel('Feature 1')
# ax.set_ylabel('Feature 2')
# ax.set_zlabel('Feature 3')
#
# plt.legend()
# plt.show()
# print('query_label.shape')
# print(query_label.shape)
# 获取test支持集，查询集，查询集标签

N_num_test = 400
pos_num_test = 126
neg_num_test = N_num_test - pos_num_test  # 多
support_pos_sample = 64  # 64
support_neg_sample = 64  # 64
support_num_test = support_pos_sample + support_neg_sample
query_num_test = N_num_test - support_num_test
query_pos = pos_num_test - support_pos_sample
query_neg = neg_num_test - support_neg_sample

support_input_test, query_input_test, query_label_test = Generate_test(X_sorted_test)

test_query_input, test_query_label = Generate_test_pos(X_sorted_test)
# print('test_query_input.shape')
# print(test_query_input.shape)
# print('test_query_input')
# print(test_query_input)
# print('test_query_label.shape')
# print(test_query_label.shape)
# print('test_query_label')
# print(test_query_label)
#
# print('query_input_test.shape[0]')
# print(query_input_test.shape[0])


def generate_test_neg(features):
    N = y_test.sum()
    test_neg_input = torch.zeros(size=[y_test.shape[0] - N, features.shape[1]])
    for i in range(y_test.shape[0] - N):
        test_neg_input[i, :] = features[i + N, :]
    result = test_neg_input
    return result

test_neg_input = generate_test_neg(X_sorted_test)

# 获取test查询集标签
# ones = torch.ones(126-64, dtype=torch.long)
# zeros = torch.zeros(500-126-64, dtype=torch.long)
# query_label_test = torch.cat((ones, zeros)).float().requires_grad_(True)

prototype_vectors_history = []

# 参数设置
input_dim = X_sorted.shape[1]
hidden_dim = 25
output_dim = 10
n_class = 2
lr = 0.001  # 10000/1000 Ir=0.0005 epochs=1000 #acc0.8567 hidden_dim = 8 weight_decay=5e-4
epochs = 150  # 10000/1000 0.0005 3000 #acc0.5321


# 10000/1000 0.001 3000 #acc不好
# 10000/1000 0.001 1000 #acc0.1不好
# 10000/1000 0.0008 1000 #局部最优
# 30000/1000 Ir=0.0005 epochs=1000 #acc0.7913 hidden_dim = 12 weight_decay=5e-4
# 20000/1000 Ir=0.0005 epochs=1000 #acc0.8807 hidden_dim = 8 weight_decay=1e-4
# 20000/1000 Ir=0.0008 epochs=1000 #acc0.8876 hidden_dim = 8 weight_decay=1e-4 网络从简

# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.input_dim = input_dim
#         # 编码器
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 20),
#             nn.ReLU(),
#             nn.Linear(20, 12),
#             nn.ReLU(),
#             nn.Linear(12, 6),
#             nn.ReLU(),
#             nn.Linear(6, 2)
#         )
#         # 解码器
#         self.decoder = nn.Sequential(
#             nn.Linear(2, 6),
#             nn.ReLU(),
#             nn.Linear(6, 12),
#             nn.ReLU(),
#             nn.Linear(12, 20),
#             nn.ReLU(),
#             nn.Linear(20, input_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x


autoencoder = Autoencoder()

# 生成支持集和查询集
class protonet_Model(nn.Module):
    last_prototype_vectors = None
    query_embeddings_labels = []

    def __init__(self, input_dim, hidden_dim, output_dim, n_class):
        super(protonet_Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_class = n_class
        self.autoencoder = Autoencoder()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)
        self.hidden_layer_1 = nn.Linear(hidden_dim, 40)  # 20
        self.bn_hidden_1 = nn.BatchNorm1d(40)
        self.hidden_layer_2 = nn.Linear(40, 25)  # 20，15
        self.bn_hidden_2 = nn.BatchNorm1d(25)
        self.output_layer = nn.Linear(25, output_dim)  # 15
        self.dropout = nn.Dropout(p=0.3)
        self.dropout_1 = nn.Dropout(p=0.5)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # Leaky ReLU activation function
        self.relu = nn.ReLU()

        # 残差连接的线性变换层，用于将输入层与第二个隐藏层的尺寸匹配
        self.residual = nn.Linear(17, 15)

    def embedding(self, x):
        x = self.autoencoder(x)

        x = self.leaky_relu(self.bn_input(self.input_layer(x)))
        x = self.dropout_1(x)
        x = self.leaky_relu(self.bn_hidden_1(self.hidden_layer_1(x)))
        x = self.dropout_1(x)
        x = self.leaky_relu(self.bn_hidden_2(self.hidden_layer_2(x)))
        x = self.dropout_1(x)
        x = self.output_layer(x)
        return x

    def eucli_tensor(self, x, y):
        # 度量欧拉距离
        # x and y should have the same shape (2, hidden_dim)
        dist_x = torch.norm(x[0] - y[0])
        dist_y = torch.norm(x[1] - y[1])
        return torch.tensor([dist_x, dist_y])

    def forward(self, support_input, query_input, query_label):

        support_embedding = self.embedding(support_input)
        query_embedding = self.embedding(query_input)
        support_size = support_embedding.shape[0]  # 支持集的例子数目128
        every_class_num = support_size // self.num_class  # 一半为故障例，一半为正常例子
        class_meta_dict = {}  # 用于储存所有类的所有向量
        for i in range(0, self.num_class):  # 两类
            class_meta_dict[i] = torch.sum(support_embedding[i * every_class_num:(i + 1) * every_class_num, :],
                                           dim=0) / every_class_num  # 求每一类的原型向量
        prototype_vector = torch.zeros(size=[len(class_meta_dict), support_embedding.shape[1]])  # 大小：（2类数，4线性转换）
        #         prototype_vector = prototype_vector.to(device)
        for key, item in class_meta_dict.items():
            prototype_vector[key, :] = class_meta_dict[key]  # 原型向量：两行4列，第一行是故障对应原型向量，第二行是特征对应原型向量

        N_query = query_embedding.shape[0]  # 查询集的例子数目372
        result = torch.zeros(size=[N_query, self.num_class])  # 用于存储每个查询集相对于两类原型向量余弦相似度
        for i in range(0, N_query):
            query_vector = query_embedding[i].repeat(self.num_class, 1)  # 查询集嵌入到原型空间表示，并重复两次
            cosine_value = torch.cosine_similarity(prototype_vector, query_vector, dim=1)  # 求余弦相似度
            result[i] = cosine_value  # 使用余弦相似度求原型向量与查询向量之间关系eucli_value = self.eucli_tensor(prototype_vector,temp_value)
            # result = F.log_softmax(result, dim=1)#求完欧氏距离后转换成负对数

        protonet_Model.last_prototype_vectors = prototype_vector.detach().cpu().numpy()
        # protonet_Model.query_embeddings_labels = list(zip(query_embedding.detach().cpu().numpy(), query_label.detach().cpu().numpy()))
        protonet_Model.query_embeddings_labels = list(
            zip(query_embedding.detach().cpu().numpy(), query_label.detach().cpu().numpy()))

        return result

model = protonet_Model(X_sorted.shape[1], hidden_dim, output_dim, n_class)  # 模型初始化
# model.cuda()
# model.train()
# q:X_sorted的维数是多少
# a:X_sorted的维数是(248,10)
optimer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # 优化器初始化5e-4

# 定义数据集类
class YourDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # 实现获取数据的逻辑
        return self.data[index]

    def __len__(self):
        # 返回数据集大小
        return len(self.data)


# 准备用于验证的数据集
valid_data = X_sorted_test[0:1024, :]
print(valid_data)
print(valid_data.shape)

# 定义用于验证的 DataLoader
valid_loader = DataLoader(
    dataset=YourDataset(valid_data),
    batch_size=512,
    shuffle=True,
    num_workers=0
)

# 定义评价指标
criterion = nn.CrossEntropyLoss()

# 加载已训练好的模型
# model = fewModel()
# model.load_state_dict(torch.load("your_model_path"))
# model = fewModel(X_sorted_test.shape[1], hidden_dim, n_class) #模型初始化
# 将模型切换到验证模式
model.eval()


# 定义验证函数
def validate(model, valid_loader, criterion):
    # 定义变量来记录模型的总损失和准确率
    valid_loss = 0.0
    valid_acc = 0.0

    # 创建列表来存储所有批次的y_true和y_pred
    all_labels = []
    all_preds = []

    # 关闭梯度计算
    with torch.no_grad():
        for batch in valid_loader:
            # 获取数据和标签
            data = batch
            support_input_test, query_input_test, query_label_test = Generate_test(X_sorted_test)
            labels = query_label_test
            # 将数据和标签送入模型进行预测
            outputs = model.forward(support_input_test, query_input_test, query_label_test)
            # 计算预测的类别
            preds = torch.argmax(-outputs, dim=1)

            # 将这个批次的y_true和y_pred添加到列表中
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())

            # 计算准确率
            valid_acc += (preds == labels).float().mean()

    # 计算平均准确率
    valid_acc /= len(valid_loader)

    # 返回所有批次的y_true和y_pred，以及平均准确率
    return all_labels, all_preds, valid_acc


def train(epoch, support_input, query_input, query_label):
    optimer.zero_grad()
    #     support_input = support_input.to(device)
    #     query_input = query_input.to(device)
    #     query_label = query_label.to(device)
    output = model.forward(support_input, query_input, query_label)
    # print(output)
    # print(output.shape)
    # print(output.shape)
    # output = output.clone().detach().requires_grad_(True)
    output = F.log_softmax(output, dim=1)  # 求负对数
    # print(support_input, query_input)
    # print(support_input.shape, query_input.shape)
    # print(output)#output
    loss = F.nll_loss(output, query_label.long())  # loss = F.nll_loss(output, query_label)
    # print(query_label)
    print(loss)
    loss.backward()
    optimer.step()
    print("Epoch: {:04d}".format(epoch), "loss:{:.4f}".format(loss))
    return loss




valid_accs = []
losses = []  # Add a list to store the losses

from sklearn.metrics import precision_score, recall_score, f1_score

precisions = []
recalls = []
f1s = []

for i in range(epochs):
    model.train()
    support_input, query_input, query_label = Generate(X_sorted)
    loss = train(i, support_input, query_input, query_label)
    losses.append(loss.item())
    model.eval()
    y_true, y_pred, valid_acc = validate(model, valid_loader, criterion)
    print("Epoch: {:04d}".format(i), "valid_acc:{:.4f}".format(valid_acc))
    valid_accs.append(valid_acc)
    prototype_vectors_history.append(model.last_prototype_vectors)

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    # 将这些指标值添加到相应的列表中
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D

fig, ax1 = plt.subplots()
import matplotlib.pyplot as plt

# 定义一个新的图像
fig, ax1 = plt.subplots()

# 绘制损失曲线
color = 'tab:blue'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color=color)
ax1.plot(range(epochs), losses, color=color, linestyle='-', label='Loss')
ax1.tick_params(axis='y', labelcolor=color)

# 绘制准确率曲线
color = 'tab:red'
ax1.plot(range(epochs), valid_accs, color=color, linestyle='--', label='Accuracy')
ax1.tick_params(axis='y')

# 绘制精确度、召回率和F1分数的曲线
ax1.plot(range(epochs), precisions, color='tab:green', linestyle='-.', label='Precision')
ax1.plot(range(epochs), recalls, color='tab:orange', linestyle=':', label='Recall')
ax1.plot(range(epochs), f1s, color='tab:purple', linestyle='-', label='F1 Score')

# 添加图例
ax1.legend(loc='upper left')

fig.tight_layout()
plt.title('Metrics vs. Epochs')
plt.grid(True)
plt.show()

# # 绘制准确率曲线
# color = 'tab:red'
# ax1.set_xlabel('Epochs')
# ax1.set_ylabel('Validation Accuracy', color=color)
# ax1.plot(range(epochs), valid_accs, color=color)
# ax1.tick_params(axis='y', labelcolor=color)
#
# # 创建次坐标轴，共享x轴
# ax2 = ax1.twinx()
#
# # 绘制损失曲线
# color = 'tab:blue'
# ax2.set_ylabel('Loss', color=color)
# ax2.plot(range(epochs), losses, color=color)
# ax2.tick_params(axis='y', labelcolor=color)
#
# fig.tight_layout()  # 为了使标签不重叠
# plt.title('Validation Accuracy and Loss vs. Epochs')
# plt.grid(True)
# plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制原型向量轨迹
colors = ['b', 'r']
for proto_idx, color in enumerate(colors):
    xs = [v[proto_idx, 0] for v in prototype_vectors_history]
    ys = [v[proto_idx, 1] for v in prototype_vectors_history]
    zs = [v[proto_idx, 2] for v in prototype_vectors_history]

    ax.plot(xs, ys, zs, label=f"Prototype {proto_idx} Trajectory", color=color, alpha=0.5)


prototype_vectors = model.last_prototype_vectors
prototype_labels = [0, 1]

# 绘制原型向量
for i, label in enumerate(prototype_labels):
    ax.scatter(prototype_vectors[i, 0], prototype_vectors[i, 1], prototype_vectors[i, 2], label=f"Prototype {label}", marker='o', s=150, edgecolors='k')

# 绘制查询嵌入

# 互换 0 和 1
query_embeddings_labels = [(embedding, 1-label) for embedding, label in model.query_embeddings_labels]

legend_added = defaultdict(bool)

# 绘制查询嵌入
for i, (embedding, label) in enumerate(query_embeddings_labels):
    marker = '+' if label == 0 else 'x'
    color = 'b' if label == 0 else 'r'
    if not legend_added[label]:
        ax.scatter(embedding[0], embedding[1], embedding[2], label=f"Query {label}", marker=marker, c=color, alpha=0.5)
        legend_added[label] = True
    else:
        ax.scatter(embedding[0], embedding[1], embedding[2], marker=marker, c=color, alpha=0.5)


from matplotlib.ticker import FormatStrFormatter

# 将坐标轴格式化为小数点后两位
formatter = FormatStrFormatter('%.3f')
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
ax.zaxis.set_major_formatter(formatter)
ax.set_xlabel('Embedding Dimension 1')
ax.set_ylabel('Embedding Dimension 2')
ax.set_zlabel('Embedding Dimension 3')
plt.title('Prototype Vectors and Query Embeddings in 3D')
plt.legend()
plt.show()



# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# colors = ['r', 'b']
#
# for proto_idx, color in enumerate(colors):
#     xs = [v[proto_idx, 0] for v in prototype_vectors_history]
#     ys = [v[proto_idx, 1] for v in prototype_vectors_history]
#     zs = [v[proto_idx, 2] for v in prototype_vectors_history]
#
#     ax.plot(xs, ys, zs, label=f"Prototype {proto_idx} Trajectory", color=color, alpha=0.5)
#
# ax.set_xlabel('Embedding Dimension 1')
# ax.set_ylabel('Embedding Dimension 2')
# ax.set_zlabel('Embedding Dimension 3')
# plt.title('Prototype Vectors Trajectories in 3D')
# plt.legend()
# plt.show()

# query_embeddings_labels = model.query_embeddings_labels
# for i, (embedding, label) in enumerate(query_embeddings_labels):
#     # ax.scatter(embedding[0], embedding[1], embedding[2], label=f"Query {label}" if i == 0 else None, marker='x' if label == 0 else '+', c='r' if label == 0 else 'b', alpha=0.5)
#     ax.scatter(embedding[0], embedding[1], embedding[2],
#                label=f"Query {label}" if f"Query {label}" not in ax.get_legend_handles_labels()[1] else None,
#                marker='x' if label == 0 else '+', c='r' if label == 0 else 'b', alpha=0.5)

# import matplotlib.pyplot as plt
#
# # 提取原型向量及其标签
# prototype_vectors = model.last_prototype_vectors
# prototype_labels = [0, 1]  # 假设有两个原型向量，第一个为负例（标签0），第二个为正例（标签1）
#
# # 绘制原型向量
# for i, label in enumerate(prototype_labels):
#     plt.scatter(prototype_vectors[i, 0], prototype_vectors[i, 1], label=f"Prototype {label}", marker='o', s=200, edgecolors='k')
#
# # 绘制查询嵌入
# query_embeddings_labels = model.query_embeddings_labels
# for i, (embedding, label) in enumerate(query_embeddings_labels):
#     plt.scatter(embedding[0], embedding[1], label=f"Query {label}" if i == 0 else None, marker='x' if label == 0 else '+', c='r' if label == 0 else 'b', alpha=0.5)
#
# plt.xlabel('Embedding Dimension 1')
# plt.ylabel('Embedding Dimension 2')
# plt.title('Prototype Vectors and Query Embeddings')
# plt.legend()
# plt.show()
#
# # Create a figure and axes
# fig, ax = plt.subplots(figsize=(8, 6))
#
# # Normalize losses to the range of valid_accs (0-1)
# normalized_losses = [l / max(losses) for l in losses]
#
# # Plot the losses curve
# ax.plot(range(epochs), normalized_losses, label='Normalized Loss', linestyle='--')
# ax.set_xlabel('Epochs')
# ax.set_ylabel('Normalized Loss')
#
# # Plot the valid_accs curve
# ax.plot(range(epochs), valid_accs, label='Validation Accuracy', linestyle='-')
# ax.set_ylabel('Validation Accuracy')
#
# # Set the title and show the legend
# ax.set_title('Normalized Loss and Validation Accuracy vs. Epochs')
# ax.legend()
#
# # Show the grid and the plot
# ax.grid(True)
# plt.show()

# valid_accs = []
# for i in range(epochs):
#     model.train()
#     support_input, query_input, query_label = Generate(X_sorted)
#     # print('support_input, query_input, query_label')
#     # print(support_input)
#     # print(query_input)
#     # print(query_label)
#     loss = train(i, support_input, query_input, query_label)
#     model.eval()
#     valid_acc = validate(model, valid_loader, criterion)
#     print("Epoch: {:04d}".format(i), "valid_acc:{:.4f}".format(valid_acc))
#     valid_accs.append(valid_acc)

# import matplotlib.pyplot as plt
# plt.plot(range(epochs), valid_accs)
# plt.xlabel('Epochs')
# plt.ylabel('Validation Accuracy')
# plt.title('Validation Accuracy vs. Epochs')
# plt.grid(True)
# plt.show()
# # 定义数据集类
# class YourDataset(torch.utils.data.Dataset):
#     def __init__(self, data):
#         self.data = data
#
#     def __getitem__(self, index):
#         # 实现获取数据的逻辑
#         return self.data[index]
#
#     def __len__(self):
#         # 返回数据集大小
#         return len(self.data)
#
#
# # 准备用于验证的数据集
# valid_data = X_sorted_test[0:1024, :]
# print(valid_data)
# print(valid_data.shape)
#
# # 定义用于验证的 DataLoader
# valid_loader = DataLoader(
#     dataset=YourDataset(valid_data),
#     batch_size=512,
#     shuffle=True,
#     num_workers=0
# )
#
# # 定义评价指标
# criterion = nn.CrossEntropyLoss()
#
# # 加载已训练好的模型
# # model = fewModel()
# # model.load_state_dict(torch.load("your_model_path"))
# # model = fewModel(X_sorted_test.shape[1], hidden_dim, n_class) #模型初始化
# # 将模型切换到验证模式
# model.eval()
#
#
# # 定义验证函数
# def validate(model, valid_loader, criterion):
#     # 定义变量来记录模型的总损失和准确率
#     valid_loss = 0.0
#     valid_acc = 0.0
#
#     # 关闭梯度计算
#     with torch.no_grad():
#         for batch in valid_loader:
#             # 获取数据和标签
#             data = batch
#             support_input_test, query_input_test, query_label_test = Generate_test(X_sorted_test)
#             labels = query_label_test
#             #             labels = labels.to(device)
#             # 将数据和标签送入模型进行预测
#             outputs = model.forward(support_input_test, query_input_test)
#             #             outputs = outputs.to(device)
#             #outputs = -F.log_softmax(outputs, dim=1) #求负对数
#             # print(outputs)
#             # 计算损失和准确率
#             # loss = F.nll_loss(outputs, labels.long())#criterion(outputs, labels)
#             # valid_loss += loss.item()
#
#             preds = torch.argmax(-outputs, dim=1)
#             # print(preds)
#             # print(len(preds))
#
#             valid_acc += (preds == labels).float().mean()
#             # q：valid_acc += (preds == labels).float().mean()解释这段代码
#             # a：preds == labels是一个bool类型的tensor，preds == labels的维度是[32,1]，preds == labels的实例是tensor([[True],
#
#     # 计算平均损失和准确率
#     # valid_loss /= len(valid_loader)
#     valid_acc /= len(valid_loader)
#     # print(valid_acc)
#
#     return valid_acc


# 执行验证函数
# valid_acc = validate(model, valid_loader, criterion)

# 打印验证结果
print("Validation accuracy: {:.4f}".format(valid_acc))
# print("Validation loss: {:.4f}".format(valid_loss))
# 保存模型
prototype_model = model
torch.save(prototype_model, "prototype_model_method.pth")
