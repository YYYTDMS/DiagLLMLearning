import torch
import numpy as np
from tqdm import *
from joblib import load
from datetime import datetime
from pyhealth.tokenizer import Tokenizer
from pyhealth.datasets import  MIMIC4Dataset, MIMIC3Dataset
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.medcode import InnerMap
import gc
from sklearn.metrics import f1_score

def get_label_tokenizer(label_tokens):
    special_tokens = []
    label_tokenizer = Tokenizer(
        label_tokens,
        special_tokens=special_tokens,
    )
    return label_tokenizer

def batch_to_multihot(label, num_labels: int) -> torch.tensor:

    multihot = torch.zeros((len(label), num_labels))
    for i, l in enumerate(label):
        multihot[i, l] = 1
    return multihot

def prepare_labels(
        labels,
        label_tokenizer: Tokenizer,
    ) -> torch.Tensor:
    labels_index = label_tokenizer.batch_encode_2d(
        labels, padding=False, truncation=False
    )
    num_labels = label_tokenizer.get_vocabulary_size()
    labels = batch_to_multihot(labels_index, num_labels)
    return labels

def parse_datetimes(datetime_strings):
    # print(datetime_strings)
    return [datetime.strptime(dt_str, "%Y-%m-%d %H:%M") for dt_str in datetime_strings]

def timedelta_to_str(tdelta):
    days = tdelta.days
    seconds = tdelta.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return days * 1440 + hours * 60 + minutes

def convert_to_relative_time(datetime_strings):
    datetimes = parse_datetimes(datetime_strings)
    base_time = min(datetimes)
    return [timedelta_to_str(dt - base_time) for dt in datetimes]

def load_dataset(dataset, root , tables=["diagnoses_icd", "procedures_icd", "prescriptions"], task_fn = None, dev = False):
    if dataset=='mimic3':
        dataset = MIMIC3Dataset(
            root = root,
            dev = dev,
            tables = ['DIAGNOSES_ICD', 'PROCEDURES_ICD', 'PRESCRIPTIONS'], 
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
            refresh_cache=False,
        )
    elif dataset == 'mimic4':
        dataset = MIMIC4Dataset(
            root=root,
            dev=dev,
            tables=tables, 
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
            refresh_cache=False,
        )
    else:
        return load(root)
    print(len(dataset.patients))
    return dataset.set_task(task_fn=task_fn)

def get_init_tokenizers(task_dataset, keys = ['cond_hist']):
    Tokenizers = {key: Tokenizer(tokens=task_dataset.get_all_tokens(key), special_tokens=["<pad>"]) for key in keys}
    return Tokenizers

def get_parent_tokenizers(task_dataset, keys = ['cond_hist', 'procedures']):
    parent_tokenizers = {}
    dictionary = {'cond_hist':InnerMap.load("ICD9CM"), 'procedures':InnerMap.load("ICD9PROC")}
    for feature_key in keys:
        assert feature_key in dictionary.keys()
        tokens = task_dataset.get_all_tokens(feature_key)
        parent_tokens = set()
        for token in tokens:
            try:
                parent_tokens.update(dictionary[feature_key].get_ancestors(token))
            except:
                continue
        parent_tokenizers[feature_key + '_parent'] = Tokenizer(tokens=list(parent_tokens), special_tokens=["<pad>"])
    return parent_tokenizers

def split_dataset(dataset, train_ratio=0.75, valid_ratio=0.1, test_ratio=0.15):
    # Ensure the ratios sum to 1
    total = train_ratio + valid_ratio + test_ratio
    if total != 1.0:
        raise ValueError("Ratios must sum to 1.")
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)
    test_size = total_size - train_size - valid_size

    # Randomly splitting the dataset
    train_set, valid_set, test_set = random_split(dataset, [train_size, valid_size, test_size])
    return train_set, valid_set, test_set


def custom_collate_fn(batch):
    sequence_data_list = [item[0] for item in batch]
    graph_data_list = [item[1] for item in batch]
    idx_list = [item[2] for item in batch]

    sequence_data_batch = {key: [d[key] for d in sequence_data_list if d[key]!=[]] for key in sequence_data_list[0]}

    graph_data_batch = graph_data_list

    return sequence_data_batch, graph_data_batch, idx_list


def mm_dataloader(trainset, validset, testset, batch_size = 64):
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
 
    return train_loader, val_loader, test_loader 



def seq_dataloader(dataset, split_ratio = [0.1, 0.7, 0.2], batch_size = 64):
    train_dataset, val_dataset, test_dataset = split_by_patient(dataset, split_ratio)
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader 

def weight_F1(labels, predicts):
    top_ks = [10, 20, 30]
    total_f1 = []

    y_true = labels
    num_labels = labels.shape[1]

    for k in top_ks:
        indices = predicts[:, :k]
        binary_matrix = np.zeros((indices.shape[0], num_labels), dtype=int)

        # 使用高级索引快速赋值
        rows = np.repeat(np.arange(len(indices)), indices.shape[1])
        cols = indices.flatten()
        valid_cols = cols[cols < num_labels]  # 确保列索引在范围内
        valid_rows = rows[cols < num_labels]
        binary_matrix[valid_rows, valid_cols] = 1

        y_pred = binary_matrix
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        total_f1.append(weighted_f1)

    return total_f1

def code_level(labels, predicts):
    labels = np.array(labels)
    total_labels = np.where(labels == 1)[0].shape[0]
    top_ks = [10,20,30]
    total_correct_preds = []
    for k in top_ks:
        correct_preds = 0
        for i, pred in enumerate(predicts):
            index = np.argsort(-pred)[:k]
            for ind in index:
                if labels[i][ind] == 1:
                    correct_preds = correct_preds + 1
        total_correct_preds.append(float(correct_preds))

    total_correct_preds = np.array(total_correct_preds) / total_labels
    return total_correct_preds

def visit_level(labels, predicts):
    labels = np.array(labels)
    predicts = np.array(predicts)
    top_ks = [10,20,30]
    precision_at_ks = []
    for k in top_ks:
        precision_per_patient = []
        for i in range(len(labels)):
            actual_positives = np.sum(labels[i])
            denominator = min(k, actual_positives)
            top_k_indices = np.argsort(-predicts[i])[:k]
            true_positives = np.sum(labels[i][top_k_indices])
            precision = true_positives / denominator if denominator > 0 else 0
            precision_per_patient.append(precision)
        average_precision = np.mean(precision_per_patient)
        precision_at_ks.append(average_precision)
    return precision_at_ks

def recall(labels, predicts):
    labels = np.array(labels)
    predicts = np.array(predicts)
    top_ks = [10,20,30]
    precision_at_ks = []
    for k in top_ks:
        precision_per_patient = []
        for i in range(len(labels)):
            actual_positives = np.sum(labels[i])
            denominator = actual_positives
            top_k_indices = np.argsort(-predicts[i])[:k]
            true_positives = np.sum(labels[i][top_k_indices])
            precision = true_positives / denominator if denominator > 0 else 0
            precision_per_patient.append(precision)
        average_precision = np.mean(precision_per_patient)
        precision_at_ks.append(average_precision)
    return precision_at_ks

from collections import Counter
def x_level(labels, predicts, label_tokenizer):
    labels = np.array(labels)
    predicts = np.array(predicts)
    top_ks = [10, 20, 30]
    precision_at_ks = []
    for k in top_ks:
        precision_per_patient = []
        for i in range(len(labels)):
            flag = 0
            parents = []
            indices = np.where(labels[i] == 1)
            for x in indices[0]:
                name = label_tokenizer.vocabulary.idx2token[x]
                #print(name)
                if len(name) < 5:
                    flag = 1
                    break
                parents.append(name[:3])
            if flag:
                continue
            #print(parents)

            actual_positives = np.sum(labels[i])
            denominator = min(k, actual_positives)
            top_k_indices = np.argsort(-predicts[i])[:k]
            #print(top_k_indices)

            childs = []
            for x in top_k_indices:
                #print(x)
                name = label_tokenizer.vocabulary.idx2token[x]
                childs.append(name[:3])
            #print(childs)

            counter1 = Counter(parents)
            counter2 = Counter(childs)
            common_counts = counter1 & counter2
            true_positives = sum(common_counts.values())

            precision = true_positives / denominator if denominator > 0 else 0
            precision_per_patient.append(precision)
        average_precision = np.mean(precision_per_patient)
        precision_at_ks.append(average_precision)
    return precision_at_ks

def train(data_loader, model, label_tokenizer, optimizer, device):
    train_loss = 0

    for index, data in enumerate(data_loader):
        model.train()
        optimizer.zero_grad()

        if type(data)==dict:
            label = prepare_labels(data['conditions'],label_tokenizer).to(device)
        else:
            label = prepare_labels(data[0]['conditions'],label_tokenizer).to(device)
        out = model(data)
        loss = F.binary_cross_entropy_with_logits(out,label)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        del loss, label, out
        torch.cuda.empty_cache()

    return train_loss

def multi_hot_to_index(labels):
    """
    labels: FloatTensor [B, C], multi-hot
    return: LongTensor [B, C], index target for F.multilabel_margin_loss
    """
    B, C = labels.shape
    targets = torch.full((B, C), -1, dtype=torch.long, device=labels.device)
    for i in range(B):
        pos = labels[i].nonzero(as_tuple=False).view(-1)  # 正类索引
        targets[i, :len(pos)] = pos
    return targets


def train_margin_loss(data_loader, model, label_tokenizer, optimizer, device, hyper=0.05):
    train_loss = 0

    for index, data in enumerate(data_loader):
        model.train()
        optimizer.zero_grad()

        if type(data)==dict:
            label = prepare_labels(data['conditions'],label_tokenizer).to(device)
        else:
            label = prepare_labels(data[0]['conditions'],label_tokenizer).to(device)
        out = model(data)
        loss = F.binary_cross_entropy_with_logits(out,label)


        target_index = multi_hot_to_index(label)  # 需要实现这个函数
        loss_margin = F.multilabel_margin_loss(torch.sigmoid(out), target_index)
        loss=loss+ hyper*loss_margin
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        del loss, label, out
        torch.cuda.empty_cache()

    return train_loss

def valid(data_loader, model, label_tokenizer, device):
    val_loss= 0
    with torch.no_grad():
        for data in data_loader:
            model.eval()
            if type(data)==dict:    
                label = prepare_labels(data['conditions'],label_tokenizer).to(device)
            else:
                label = prepare_labels(data[0]['conditions'],label_tokenizer).to(device)   
            out = model(data)
            loss = F.binary_cross_entropy_with_logits(out,label)
            val_loss += loss.detach().cpu().numpy()
    return val_loss

def test(data_loader, model, label_tokenizer):
    y_t_all, y_p_all = [], []

    with torch.no_grad():
        for data in tqdm(data_loader):
            model.eval()
            if type(data)==dict:
                label = prepare_labels(data['conditions'],label_tokenizer)
            else:
                label = prepare_labels(data[0]['conditions'],label_tokenizer)

            out = model(data)


            y_t = label.cpu().numpy()
            y_p = torch.sigmoid(out).detach().cpu().numpy()
            y_t_all.append(y_t)
            y_p_all.append(y_p)
        y_true = np.concatenate(y_t_all, axis=0)
        y_prob = np.concatenate(y_p_all, axis=0)

    return y_true, y_prob
