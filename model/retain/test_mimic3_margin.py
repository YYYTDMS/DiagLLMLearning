import os
from tqdm import *
import random
import argparse

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
ckptpath = "./saves/{}_margin_trained_{}_{}_time_box_dataset_1.ckpt".format(hyper,
    args.model, args.dataset
)

best_model = torch.load(ckptpath)
model.load_state_dict(best_model)
model = model.to(device)




y_true, y_prob = test(test_loader, model, label_tokenizer)

np.save(f"{hyper}margin_{args.dataset}_{args.model}_y_true.npy", y_true)
np.save(f"{hyper}margin_{args.dataset}_{args.model}_y_prob.npy", y_prob)

print("vcfr")

print(visit_level(y_true, y_prob))
print(code_level(y_true, y_prob))

pred = np.argsort(-y_prob)
print(weight_F1(y_true, pred))

print(recall(y_true, y_prob))
