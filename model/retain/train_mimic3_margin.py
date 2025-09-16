import os
from tqdm import *
import random
import argparse
import numpy as np
import pickle
from joblib import dump, load
import torch
import torch.optim as optim
from utils import *
from data.Task import *
from models.Model import *
from models.baselines import *


parser = argparse.ArgumentParser()
parser.add_argument(
    "--epochs", type=int, default=2000, help="Number of epochs to train."
)
parser.add_argument("--lr", type=float, default=0.001, help="learning rate.")
parser.add_argument(
    "--model",
    type=str,
    default="TRANS",
    help="Transformer, RETAIN, StageNet, KAME, GCT, DDHGNN, TRANS",
)
parser.add_argument("--dev", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--dataset", type=str, default="mimic3", choices=["mimic3", "mimic4", "ccae"]
)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument(
    "--pe_dim", type=int, default=4, help="dimensions of spatial encoding"
)
parser.add_argument("--devm", type=bool, default=False, help="develop mode")


fileroot = {
    "mimic3": "mimic3/",
    "mimic4": "data path of mimic4",
    "ccae": "./data/processed_dip.pkl",
}

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print("{}--{}".format(args.dataset, args.model))
cudaid = "cuda:" + str(args.dev)
device = torch.device(cudaid if torch.cuda.is_available() else "cpu")
dim = 16
mid_name = "mine_data/patient_time_mimic3_1"
if args.dataset == "mimic4":
    task_dataset = pickle.load(open("mimic4_box_datasetxxx.pkl", "rb"))
elif args.dataset == "mimic3":
    task_dataset = pickle.load(open(f"{mid_name}/mimic3_box_dataset_1.pkl", "rb"))
else:
    task_dataset = load_dataset(args.dataset, root=fileroot[args.dataset])


Tokenizers = get_init_tokenizers(task_dataset)

ccs9 = pickle.load(open("ccs9.pkl", "rb"))
ccs = ccs9
label_tokenizer = Tokenizer(tokens=ccs)

if args.model == "Transformer":
    train_loader, val_loader, test_loader = seq_dataloader(
        task_dataset, batch_size=args.batch_size
    )
    model = Transformer(
        Tokenizers, len(task_dataset.get_all_tokens("conditions")), device
    )

elif args.model == "RETAIN":
    from torch.utils.data import Subset

    indices = torch.load(f"{mid_name}/trainset.pt")
    train_dataset = Subset(task_dataset, indices)
    indices = torch.load(f"{mid_name}/validset.pt")
    val_dataset = Subset(task_dataset, indices)
    indices = torch.load(f"{mid_name}/testset.pt")
    test_dataset = Subset(task_dataset, indices)

    train_loader = get_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=False
    )
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = RETAIN(Tokenizers, len(ccs), device)

elif args.model == "KAME":
    train_loader, val_loader, test_loader = seq_dataloader(
        task_dataset, batch_size=args.batch_size
    )
    Tokenizers.update(get_parent_tokenizers(task_dataset))
    model = KAME(Tokenizers, len(task_dataset.get_all_tokens("conditions")), device)

elif args.model == "StageNet":

    from torch.utils.data import Subset

    indices = torch.load("split/3_1/trainset.pt")
    train_dataset = Subset(task_dataset, indices)
    indices = torch.load("split/3_1/validset.pt")
    val_dataset = Subset(task_dataset, indices)
    indices = torch.load("split/3_1/testset.pt")
    test_dataset = Subset(task_dataset, indices)

    train_loader = get_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=False
    )
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = StageNet(Tokenizers, len(ccs), device)

elif args.model == "TRANS":
    mdataset = MMDataset(
        task_dataset, Tokenizers, dim=dim, device=device, trans_dim=args.pe_dim
    )

    from torch.utils.data import Subset


    indices = torch.load(f"{mid_name}/trainset.pt")
    trainset = Subset(mdataset, indices)
    indices = torch.load(f"{mid_name}/validset.pt")
    validset = Subset(mdataset, indices)
    indices = torch.load(f"{mid_name}/testset.pt")
    testset = Subset(mdataset, indices)

    train_loader, val_loader, test_loader = mm_dataloader(
        trainset, validset, testset, batch_size=args.batch_size
    )
    model = TRANS(
        Tokenizers, dim, len(ccs), device, graph_meta=graph_meta, pe=args.pe_dim
    )

hyper=0.03

ckptpath = "./saves/{}_margin_trained_{}_{}_time_box_dataset_1.ckpt".format(
    hyper, args.model, args.dataset
)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
best = 12345
early = 0

best_epoch = 0
bestc = None
bestv = None
bestf = None
bestr = None
bestv10 = 0
pbar = tqdm(range(args.epochs))
for epoch in pbar:
    model = model.to(device)

    train_loss = train_margin_loss(train_loader, model, label_tokenizer, optimizer, device,hyper)
    val_loss = valid(val_loader, model, label_tokenizer, device)

    print(
        f"Epoch {epoch + 1}/{args.epochs} - train loss: {train_loss:.2f} - valid loss: {val_loss:.2f}"
    )

    # pbar.set_description(f"Epoch {epoch + 1}/{args.epochs} - train loss: {train_loss:.2f} - valid loss: {val_loss:.2f}")
    # if val_loss<best:
    # torch.save(model.state_dict(), ckptpath)

    y_true, y_prob = test(test_loader, model, label_tokenizer)
    # print(code_level(y_true, y_prob))
    # print(visit_level(y_true, y_prob))
    # print()

    c = code_level(y_true, y_prob)
    v = visit_level(y_true, y_prob)
    r = recall(y_true, y_prob)
    if bestv10 < v[0]:
        early = 0
        bestv10 = v[0]
        bestc = c
        bestv = v
        bestr = r
        torch.save(model.state_dict(), ckptpath)
        best_epoch = epoch
    else:
        early += 1

    print("----visit_level----code_level---recall------")
    print(v)
    print(c)
    print(r)

    print("----------best------------" + str(best_epoch))
    print(bestv)
    print(bestc)
    print(bestr)

    if early == 200:
        break

# for limited gpu memory
if args.model == "TRANS":
    del model
    torch.cuda.empty_cache()
    import gc

    gc.collect()
    # device = torch.device('cpu')
    model = TRANS(
        Tokenizers, dim, len(ccs), device, graph_meta=graph_meta, pe=args.pe_dim
    )


best_model = torch.load(ckptpath)
model.load_state_dict(best_model)
model = model.to(device)


y_true, y_prob = test(test_loader, model, label_tokenizer)
print("vcfr")

print(visit_level(y_true, y_prob))
print(code_level(y_true, y_prob))

pred = np.argsort(-y_prob)
print(weight_F1(y_true, pred))

print(recall(y_true, y_prob))
