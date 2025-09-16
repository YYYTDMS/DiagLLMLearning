from models.HGT import *
from models.Seqmodels import *
from layers.TSEncoder import *
import torch
import torch.nn as nn
from torch_geometric.data import Batch
import gc

feats_to_nodes = {
    'cond_hist': 'co',
    'procedures': 'pr',
    'drugs': 'dh',
    'co': 'cond_hist',
    'pr': 'procedures',
    'dh': 'drugs'
}

graph_meta = (['visit', 'co', 'pr', 'dh'],
 [('co', 'in', 'visit'),
  ('pr', 'in', 'visit'),
  ('dh', 'in', 'visit'),
  ('visit', 'connect', 'visit'),
  ('visit', 'has', 'co'),
  ('visit', 'has', 'pr'),
  ('visit', 'has', 'dh')])

class TRANS(nn.Module):
    def __init__(
        self,
        Tokenizers,
        hidden_size,
        output_size,
        device,
        graph_meta,
        embedding_dim = 16,
        dropout = 0.5,
        num_heads = 2,
        num_layers = 2,
        pe = False
    ):
        super(TRANS, self).__init__()
        self.embedding_dim = embedding_dim
        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.feature_keys = Tokenizers.keys()

        self.feature_keys = ['cond_hist']

        self.device = device

        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)

        self.transformer = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.transformer[feature_key] = TransformerLayer(
                feature_size=embedding_dim, dropout=dropout
            )
            # if feature_key == 'cond_hist':
            #     self.transformer[feature_key] = TransformerLayer(
            #         feature_size=embedding_dim, dropout=dropout
            #     )
            # else:
            #     self.transformer[feature_key] = TransformerLayer(
            #         feature_size=32, dropout=dropout
            #     )

        self.tim2vec = Time2Vec(8).to(device)
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)
        # self.fc = nn.Linear(128, output_size)

        self.graphmodel = HGT(hidden_channels = hidden_size, out_channels = output_size, num_heads=num_heads, num_layers = num_layers, metadata = graph_meta).to(device)
        self.pe = pe
        self.spatialencoder = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.spatialencoder[feature_key] = nn.Linear(self.pe*2, embedding_dim)#.to(self.device)
        self.alpha = 0.8


        # self.fc1 = nn.Linear(len(self.feature_keys) * self.embedding_dim, 128)
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

        #self.network1 = nn.Sequential(nn.Linear(len(self.feature_keys) * self.embedding_dim, 128),nn.ReLU(),nn.Linear(128, 64),nn.ReLU(),nn.Linear(64, 16))

    def set_e(self, e_best):
        self.ee = nn.Embedding.from_pretrained(e_best.clone()).to(self.device)
        self.ee.weight.requires_grad_(True)

    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]

        self.embeddings[feature_key] = nn.Embedding(
            tokenizer.get_vocabulary_size(),
            self.embedding_dim,
            padding_idx=tokenizer.get_padding_index(),
        )
        # if feature_key == 'cond_hist':
        #     self.embeddings[feature_key] = nn.Embedding(
        #             tokenizer.get_vocabulary_size(),
        #             self.embedding_dim,
        #             padding_idx=tokenizer.get_padding_index(),
        #         )
        # else:
        #     self.embeddings[feature_key] = nn.Embedding(
        #         tokenizer.get_vocabulary_size(),
        #         32,
        #         padding_idx=tokenizer.get_padding_index(),
        #     )
    def get_embedder(self):
        feature = {}
        for k in self.embeddings.keys():
            lenth = self.feat_tokenizers[k].get_vocabulary_size()
            tensor = torch.arange(0, lenth, dtype=torch.long).to(self.device)
            feature[k] = self.embeddings[k](tensor)
        return feature
        
    def process_seq(self, seqdata):
        patient_emb = []
        for feature_key in self.feature_keys:
            x = self.feat_tokenizers[feature_key].batch_encode_3d(
                seqdata[feature_key],
            )
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            x = self.embeddings[feature_key](x)
            x = torch.sum(x, dim=2)
            mask = torch.any(x !=0, dim=2)
            _, x = self.transformer[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        #patient_emb = torch.stack(patient_emb, dim=0).sum(dim=0)
        #print(patient_emb.shape)

        logits = self.fc(patient_emb)


        del x
        torch.cuda.empty_cache()

        return logits, patient_emb
    
    def process_graph_fea(self, graph_list, pe):
        f = self.get_embedder()
        for i in range(len(graph_list)):
            for node_type, x in graph_list[i].x_dict.items():
                if node_type!='visit':
                    graph_list[i][node_type].x = f[feats_to_nodes[node_type]]
                if node_type=='visit':
                    timevec = self.tim2vec(torch.tensor(graph_list[i]['visit'].time, dtype = torch.float32, device=self.device))
                    num_visit = graph_list[i]['visit'].x.shape[0]
                    graph_list[i]['visit'].x = torch.cat([pe[i].repeat(num_visit, 1), timevec],dim=-1)

        # x = Batch.from_data_list(graph_list)
        #
        # del f, pe, timevec, num_visit, graph_list, lpe, rws, se
        # torch.cuda.empty_cache()

        return Batch.from_data_list(graph_list)
        # return x
    
    def forward(self, batchdata):

        seq_logits, Patient_emb = self.process_seq(batchdata[0])

        return seq_logits

