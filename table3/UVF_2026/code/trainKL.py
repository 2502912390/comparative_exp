import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassRecall

import numpy as np
import random

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchmetrics import Precision, Recall, F1Score
from torch.utils.data import DataLoader, TensorDataset


from dataset import *
from model import *
from utils import *

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(12)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {}
batch_size = 512


dataset = Reuters3()
config['num_class'] = dataset.num_class
config['view_nums'] = len(dataset.views)
X_train, y_train, X_test, y_test = dataset.data()

input_size = [X_train[key].size(1) for key in X_train.keys()]
config['input_size'] = input_size

X_train = {key: value.to(device) for key, value in X_train.items()}
y_train = y_train.to(device)
X_test = {key: value.to(device) for key, value in X_test.items()}
y_test = y_test.to(device)

model = UVF(config).to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
best_acc = 0
best_f1 = 0
best_pre = 0
best_recall = 0

for epoch in range(epochs):
    print("------第{}轮训练------".format((epoch + 1)))

    train_datasets = {}

    for key, value in X_train.items():
        train_datasets[key] = value

    train_dataset = TensorDataset(*train_datasets.values(), y_train)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_train_batch = {}
    # for batch_idx, (data1, data2, data3, data4, data5, data6, data7, target) in enumerate(train_loader):
    for batch_idx, (data1, data2, data3, data4, data5, target) in enumerate(train_loader):
        X_train_batch['view1'] = data1
        X_train_batch['view2'] = data2
        X_train_batch['view3'] = data3
        X_train_batch['view4'] = data4
        X_train_batch['view5'] = data5
        # X_train_batch['view6'] = data6
        # X_train_batch['view7'] = data7

        optimizer.zero_grad()
        output, gate, input, selected_features = model(X_train_batch, True)
        mes_loss = 0
        for i in range(len(input)):
            mes_loss += mse_constraint(input[i], selected_features[i])
        mes_loss = mes_loss / len(input)

        l1_loss = 0
        for g in gate:
            l1_loss += torch.mean(torch.abs(g))  # 或 g.sum()


        alpha = 0  # 超参数，需要调节
        classification_loss = loss_fn(output.squeeze(), target.long())
        beta = 0
        loss = classification_loss + alpha * l1_loss / config['num_class'] + beta * mes_loss

        # loss = lambda_l1 * l1_loss + loss

        _, predicted_labels = torch.max(output.data, 1)
        correct_predictions = (predicted_labels.squeeze() == target).sum().item()

        total_samples = len(target)

        accuracy = correct_predictions / total_samples

        loss.backward()
        optimizer.step()

    # 测试
    with torch.no_grad():
        output, gate, input, selected_features = model(X_test, False)
        loss = loss_fn(output.squeeze(), y_test.long())

        _, predicted_labels = torch.max(output.data, 1)
        correct_predictions = (predicted_labels.squeeze() == y_test).sum().item()

        total_samples = len(y_test)

        accuracy = correct_predictions / total_samples


        y_test_cpu = y_test.cpu()
        predicted_labels_cpu = predicted_labels.cpu()

        precision = precision_score(y_test_cpu.numpy(), predicted_labels_cpu.numpy(), average='weighted')
        recall = recall_score(y_test_cpu.numpy(), predicted_labels_cpu.numpy(), average='weighted')
        f1 = f1_score(y_test_cpu.numpy(), predicted_labels_cpu.numpy(), average='weighted')
        if accuracy > best_acc:
            best_acc = accuracy
            best_f1 = f1
            best_pre = precision
            best_recall = recall


print("Acc:{}".format(best_acc))
print("F1:{}".format(best_f1))
print("pre:{}".format(best_pre))
print("recall:{}".format(best_recall))


