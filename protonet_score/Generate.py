import torch
import random


N_num = 4000
pos_num = 248
neg_num = N_num - pos_num  
support_pos_sample = 64 
support_neg_sample = 64 
support_num = support_pos_sample + support_neg_sample
query_num = N_num - support_num
query_pos = pos_num - support_pos_sample
query_neg = neg_num - support_neg_sample

N_num_test = 400
pos_num_test = 126
neg_num_test = N_num_test - pos_num_test 
support_pos_sample = 64  
support_neg_sample = 64  
support_num_test = support_pos_sample + support_neg_sample
query_num_test = N_num_test - support_num_test

def Generate(features):
    N = N_num  
    N_pos = pos_num  
    support_input = torch.zeros(size=[support_num, features.shape[1]])  
    query_input = torch.zeros(size=[query_num, features.shape[1]])
    postive_list = list(range(0, N_pos)) 
    negtive_list = list(range(N_pos, N)) 
    support_list_pos = random.sample(postive_list, support_pos_sample)  
    support_list_neg = random.sample(negtive_list, support_neg_sample)
    query_list_pos = [item for item in postive_list if item not in support_list_pos] 
    query_list_neg = [item for item in negtive_list if item not in support_list_neg] 
    index = 0
    for item in support_list_pos:
        support_input[index, :] = features[item, :]
        index += 1
    for item in support_list_neg:
        support_input[index, :] = features[item, :]
        index += 1
    index = 0
    for item in query_list_pos:
        query_input[index, :] = features[item, :]
        index += 1
    for item in query_list_neg:
        query_input[index, :] = features[item, :]
        index += 1
    ones = torch.ones(pos_num - support_pos_sample, dtype=torch.long)
    zeros = torch.zeros(neg_num - support_neg_sample, dtype=torch.long)
    result = torch.cat((ones, zeros)).float().requires_grad_(True)
    query_label = result
    return support_input, query_input, query_label


def Generate_test(features):
    N = N_num_test  
    N_pos = pos_num_test  
    support_input_test = torch.zeros(size=[support_num_test, features.shape[1]]) 
    query_input_test = torch.zeros(size=[query_num_test, features.shape[1]]) 
    postive_list = list(range(0, N_pos)) 
    negtive_list = list(range(N_pos, N)) 
    support_list_pos = random.sample(postive_list, support_pos_sample) 
    support_list_neg = random.sample(negtive_list, support_neg_sample)  
    query_list_pos = [item for item in postive_list if item not in support_list_pos]  
    query_list_neg = [item for item in negtive_list if item not in support_list_neg] 
    index = 0
    for item in support_list_pos:
        support_input_test[index, :] = features[item, :]
        index += 1
    for item in support_list_neg:
        support_input_test[index, :] = features[item, :]
        index += 1
    index = 0
    for item in query_list_pos:
        query_input_test[index, :] = features[item, :]
        index += 1
    for item in query_list_neg:
        query_input_test[index, :] = features[item, :]
        index += 1
    ones = torch.ones(pos_num_test - support_pos_sample, dtype=torch.long)
    zeros = torch.zeros(neg_num_test - support_neg_sample, dtype=torch.long)
    result = torch.cat((ones, zeros)).float().requires_grad_(True)
    query_label_test = result
    return support_input_test, query_input_test, query_label_test

test_pos = 126
def Generate_test_pos(features):
    N = test_pos 
    test_pos_input = torch.zeros(size=[test_pos, features.shape[1]]) 
    test_pos_list = list(range(0, N)) 
    index = 0
    for item in test_pos_list:
        test_pos_input[index, :] = features[item, :]
        index += 1
    ones = torch.ones(test_pos, dtype=torch.long)
    test_query_label = ones
    return test_pos_input, test_query_label

