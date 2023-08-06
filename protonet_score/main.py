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

X_train_scaled, y_train, X_test_scaled, y_test = load_data()

sort_index = np.argsort(-y_train)
X_sorted = X_train_scaled[sort_index]
y_sorted = y_train[sort_index]

sort_index_test = np.argsort(-y_test)
X_sorted_test = X_test_scaled[sort_index_test]
y_sorted_test = y_test[sort_index_test]


features = X_sorted

global last_prototype_vectors
global query_embeddings_labels

test_pos = 126
N_num = 4000
pos_num = 248
neg_num = N_num - pos_num 
support_pos_sample = 64 
support_neg_sample = 64  
support_num = support_pos_sample + support_neg_sample
query_num = N_num - support_num
query_pos = pos_num - support_pos_sample
query_neg = neg_num - support_neg_sample

X_sorted = torch.from_numpy(X_sorted)  
X_sorted_test = torch.from_numpy(X_sorted_test) 
y_sorted = torch.from_numpy(y_sorted) 
y_sorted_test = torch.from_numpy(y_sorted_test)

support_input, query_input, query_label = Generate(X_sorted)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

query_input_np = query_input.numpy()
query_label_np = query_label.detach().numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


N_num_test = 400
pos_num_test = 126
neg_num_test = N_num_test - pos_num_test  
support_pos_sample = 64
support_neg_sample = 64  
support_num_test = support_pos_sample + support_neg_sample
query_num_test = N_num_test - support_num_test
query_pos = pos_num_test - support_pos_sample
query_neg = neg_num_test - support_neg_sample

support_input_test, query_input_test, query_label_test = Generate_test(X_sorted_test)

test_query_input, test_query_label = Generate_test_pos(X_sorted_test)

def generate_test_neg(features):
    N = y_test.sum()
    test_neg_input = torch.zeros(size=[y_test.shape[0] - N, features.shape[1]])
    for i in range(y_test.shape[0] - N):
        test_neg_input[i, :] = features[i + N, :]
    result = test_neg_input
    return result

test_neg_input = generate_test_neg(X_sorted_test)

prototype_vectors_history = []

# 参数设置
input_dim = X_sorted.shape[1]
hidden_dim = 25
output_dim = 10
n_class = 2
lr = 0.001  
epochs = 150  

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


    def embedding(self, x):
        x = self.autoencoder(x)

        x = self.leaky_relu(self.bn_input(self.input_layer(x)))
        x = self.dropout_1(x)
        x = self.relu(self.bn_hidden_1(self.hidden_layer_1(x)))
        x = self.dropout_1(x)
        x = self.leaky_relu(self.bn_hidden_2(self.hidden_layer_2(x)))
        x = self.dropout_1(x)
        x = self.output_layer(x)
        return x

    def eucli_tensor(self, x, y):
        
        dist_x = torch.norm(x[0] - y[0])
        dist_y = torch.norm(x[1] - y[1])
        return torch.tensor([dist_x, dist_y])

    def forward(self, support_input, query_input, query_label):
        number_class = {} 
        support_embedding = self.embedding(support_input)
        query_embedding = self.embedding(query_input)
        support_size = support_embedding.shape[0]  
        counted_class = support_size // self.num_class 
        for i in range(0, self.num_class): 
            number_class[i] = torch.sum(support_embedding[i * counted_class:(i + 1) * counted_class, :],dim=0) / counted_class  
        prototype_vector = torch.zeros(size=[len(number_class), support_embedding.shape[1]])  
       
        for proto_num, item in number_class.items():
            prototype_vector[proto_num, :] = number_class[proto_num] 

        N_query = query_embedding.shape[0] 
        result = torch.zeros(size=[N_query, self.num_class])  
        for i in range(0, N_query):
            query_vector = query_embedding[i].repeat(self.num_class, 1) 
            cosine_value = torch.cosine_similarity(prototype_vector, query_vector, dim=1) 
            result[i] = cosine_value 
            
        protonet_Model.last_prototype_vectors = prototype_vector.detach().cpu().numpy()
        
        protonet_Model.query_embeddings_labels = list(
            zip(query_embedding.detach().cpu().numpy(), query_label.detach().cpu().numpy()))

        return result

model = protonet_Model(X_sorted.shape[1], hidden_dim, output_dim, n_class)  

optimer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  

# class
class YourDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):

        return len(self.data)


# prepare
valid_data = X_sorted_test[0:1024, :]
print(valid_data)
print(valid_data.shape)

# valid DataLoader
valid_loader = DataLoader(
    dataset=YourDataset(valid_data),
    batch_size=512,
    shuffle=True,
    num_workers=0
)


criterion = nn.CrossEntropyLoss()


model.eval()


# define validate
def validate(model, valid_loader, criterion):

    valid_loss = 0.0
    valid_acc = 0.0


    all_labels = []
    all_preds = []


    with torch.no_grad():
        for batch in valid_loader:

            data = batch
            support_input_test, query_input_test, query_label_test = Generate_test(X_sorted_test)
            labels = query_label_test

            outputs = model.forward(support_input_test, query_input_test, query_label_test)

            preds = torch.argmax(-outputs, dim=1)

     
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())


            valid_acc += (preds == labels).float().mean()


    valid_acc /= len(valid_loader)


    return all_labels, all_preds, valid_acc


def train(epoch, support_input, query_input, query_label):
    optimer.zero_grad()

    output = model.forward(support_input, query_input, query_label)

    output = F.log_softmax(output, dim=1)  

    loss = F.nll_loss(output, query_label.long())  

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

    # add metr
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D

fig, ax1 = plt.subplots()
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

# plot
color = 'tab:red'
ax1.plot(range(epochs), valid_accs, color=color, linestyle='--', label='Accuracy')
ax1.tick_params(axis='y')

# add
ax1.plot(range(epochs), precisions, color='tab:green', linestyle='-.', label='Precision')
ax1.plot(range(epochs), recalls, color='tab:orange', linestyle=':', label='Recall')
ax1.plot(range(epochs), f1s, color='tab:purple', linestyle='-', label='F1 Score')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


prototype_vectors = model.last_prototype_vectors
prototype_labels = [0, 1]

print("Validation accuracy: {:.4f}".format(valid_acc))


prototype_model = model
torch.save(prototype_model, "prototype_model_method.pth")




