import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import layers


class AMF(torch.nn.Module):
    def __init__(self, num_of_drugs, emb_size=256, dropout_rate=0.3):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=num_of_drugs, embedding_dim=emb_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.emb_b = nn.Embedding(num_embeddings=num_of_drugs, embedding_dim=1)
        self.out = nn.Linear(in_features=emb_size+1, out_features=1)

    def forward(self, a, b):
        a = a.cuda()
        b = b.cuda()
        emb1_b = self.emb_b(a)
        emb2_b = self.emb_b(b)
        b_add = torch.add(emb1_b, emb2_b)

        emb1 = self.emb(a)
        emb2 = self.emb(b)
        # dropout_emb1 = self.dropout(emb1)
        # dropout_emb2 = self.dropout(emb2)
        mult = torch.mul(emb1, emb2)

        concat = torch.cat((b_add, mult), 1)
        out = self.out(concat)
        out = F.sigmoid(out)

        return out

class Con_LSTM(torch.nn.Module):
    def __init__(self, num_of_drugs, emb_size=256,):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=num_of_drugs, embedding_dim=emb_size)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32,kernel_size=8, stride=2, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding='same')

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding='same')

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4, stride=1, padding='same')

        self.LSTM1 = nn.LSTM(input_size=128)
        self.LSTM2 = nn.LSTM(input_size=64)

        self.out = nn.Linear(in_features=64, out_features=1)

    def forward(self, a, b):
        a = a.cuda()
        b = b.cuda()
        a = self.emb(a)
        b = self.emb(b)

        x = torch.cat((a, b), 1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))

        x = F.dropout(self.LSTM1(x), 0.5)
        x = F.dropout(self.LSTM2(x), 0.2)

        x = torch.flatten(x)
        x = F.sigmoid(self.out(x))
        return x





class IPGAT(torch.nn.Module):
    def __init__(self, num_of_drugs, emb_size=256, atten_size=128, dropout_rate=0.3, nheads=3):
        super().__init__()
        self.emb_size = emb_size
        self.atten_size = atten_size
        self.emb = nn.Embedding(num_embeddings=num_of_drugs, embedding_dim=emb_size)
        self.dropout = dropout_rate
        self.emb_b = nn.Embedding(num_embeddings=num_of_drugs, embedding_dim=1)

        #attention
        self.attentions1 = [layers.GraphAttentionLayer(emb_size, atten_size, dropout=dropout_rate) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention)
        self.attentions2 = [layers.GraphAttentionLayer(atten_size*nheads, atten_size, dropout=dropout_rate) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention)
        # self.attentions3 = [layers.GraphAttentionLayer(atten_size * nheads, atten_size, dropout=dropout_rate) for _ in range(nheads)]
        # for i, attention in enumerate(self.attentions3):
        #     self.add_module('attention3_{}'.format(i), attention)

        self.a = nn.Parameter(torch.empty(size=(1, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.b = nn.Parameter(torch.empty(size=(1, 1)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        # self.c = nn.Parameter(torch.empty(size=(1, 1)))
        # nn.init.xavier_uniform_(self.c.data, gain=1.414)

        self.linear = nn.Linear(in_features=atten_size*nheads, out_features=atten_size)
        self.out = nn.Linear(in_features=atten_size+1, out_features=1)

    def forward(self, a, b, adj):
        a = a.cuda()
        b = b.cuda()
        nodes = np.array([i for i in range(adj.shape[0])])
        nodes = torch.LongTensor(nodes).cuda()
        # nodes = F.gelu(self.emb(nodes))
        emb = F.dropout(self.emb(nodes), self.dropout, training=self.training)

        atten1 = torch.cat([att(emb, adj) for att in self.attentions1], dim=1)
        atten2 = torch.cat([att(atten1, adj) for att in self.attentions2], dim=1)
        # atten3 = torch.cat([att(atten2, adj) for att in self.attentions3], dim=1)
        atten1 = torch.mul(atten1, self.a/(self.a+self.b))
        atten2 = torch.mul(atten2, self.b/(self.a+self.b))
        # atten3 = torch.mul(atten3, self.c/(self.a+self.b+self.c))

        atten = torch.add(atten1, atten2)
        # atten = torch.add(atten, atten3)

        at_a = torch.index_select(atten, 0, a)
        at_b = torch.index_select(atten, 0, b)
        # dropout_atten_a = F.dropout(at_a, self.dropout, training=self.training)
        # dropout_atten_b = F.dropout(at_b, self.dropout, training=self.training)
        x = torch.mul(at_a, at_b)

        x = self.linear(x)
        x = F.gelu(x)

        emb1_b = self.emb_b(a)
        emb2_b = self.emb_b(b)
        b_add = F.gelu(torch.add(emb1_b, emb2_b))
        x = torch.cat((x, b_add), 1)

        out = torch.sigmoid(self.out(x))

        return out


