import random
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader  
from torch import nn
import torch.nn.functional as F

random.seed(0)
np.random.seed(0)

word = '<PAD>,1,2,3,4,5,6,7,8,9,0,<SOS>,<EOS>,;'
vocab = {word: i for i, word in enumerate(word.split(','))} 
vocab_r = [k for k, v in vocab.items()] #反查词典  
device = 'cuda'
N_layer = 5
n_heads = 8
input_dim = 64
hidden_dim = 128


def get_data():
    num_to_be_sort = 10
    maxnum = 100
    x_list = []
    y_list = []

    for i in range(num_to_be_sort):
        x_list.append(np.random.randint(0,maxnum))
    y_list = sorted(x_list)
    

    x = ''
    y = ''

    for i in range(num_to_be_sort):
        x = x+str(x_list[i]) + ';'
        y = y+str(y_list[i]) + ';'
    x =x[:-1]
    y = y[:-1]

    x = [ch for ch in x]
    y = [ch for ch in y]

    x = ['<SOS>'] + x + ['<EOS>']
    y = ['<SOS>'] + y + ['<EOS>']

    x = x + ['<PAD>'] * 50
    y = y + ['<PAD>'] * 51
    x = x[:50]
    y = y[:51]

    token_x = [vocab[i] for i in x]
    token_y = [vocab[i] for i in y]

    tensor_x = torch.LongTensor(token_x)
    tensor_y = torch.LongTensor(token_y)

    return tensor_x.to(device), tensor_y.to(device)

def show_data(tensor_x,tensor_y):
    word_x = "".join([vocab_r[i] for i in tensor_x.tolist()])
    word_y = "".join([vocab_r[i] for i in tensor_y.tolist()])
    return word_x, word_y


# 定义数据集  
class SortDataset(torch.utils.data.Dataset):  
    def __init__(self,size = 100000):  
        super(Dataset, self).__init__()  
        self.size = size  
  
    def __len__(self):  
        return self.size  
  
    def __getitem__(self, i):  
        return get_data()  
      
ds_train = SortDataset(size = 100000)
ds_val = SortDataset(size = 10000)

# 数据加载器
dl_train = DataLoader(dataset=ds_train,
         batch_size=200,
         drop_last=True,
         shuffle=True)

dl_val = DataLoader(dataset=ds_val,
         batch_size=200,
         drop_last=True,
         shuffle=False)

## 训练时使用 传入tgt为y的前n-1个字符
def create_mask(src, tgt):
    src_mask = (src != vocab['<PAD>']).unsqueeze(-2)
    tgt_mask = (tgt != vocab['<PAD>']).unsqueeze(-2)
    tgt_len = tgt.shape[-1]
    tgt_mask = tgt_mask & torch.tril(torch.ones(1,tgt_len, tgt_len,device=device)).bool()
    return src_mask, tgt_mask

class MaskedBatch():
    def __init__(self,src,tgt,pad=0):
        self.src = src
        self.tgt = tgt[:,:-1]
        self.tgt_y = tgt[:, 1:]
        self.src_mask,self.tgt_mask = create_mask(src,tgt[:,:-1])
        self.ntoken = (self.tgt_y!= pad).sum()


class attn(nn.Module):
    def __init__(self):
        super(attn, self).__init__()

    def forward(self, Q, K, V,mask=None,dropout=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(K.size(-1))
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask != True, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return p_attn.matmul(V)

class multihead_attn(nn.Module):
    def __init__(self, n_heads, input_dim, dropout=0.1):
        super(multihead_attn, self).__init__()
        assert input_dim % n_heads == 0
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.dim_each_head = input_dim // n_heads
        self.WQ = nn.Linear(input_dim, input_dim)
        self.WK = nn.Linear(input_dim, input_dim)
        self.WV = nn.Linear(input_dim, input_dim)
        self.WO = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = attn()

    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        seq_len = Q.shape[1]
        input_dim = Q.shape[2]
        Q = self.WQ(Q)
        K = self.WK(K)
        V = self.WV(V)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.dim_each_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.dim_each_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.dim_each_head).transpose(1, 2)
        attn_score = self.attn(Q, K, V, mask,dropout=self.dropout)

        output = attn_score.transpose(1, 2).contiguous().view(batch_size, seq_len, input_dim)
        return self.WO(output)



class feedforwardLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.1):
        super(feedforwardLayer, self).__init__()
        self.function1 = nn.Linear(input_dim, hidden_dim)
        self.function2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.function2(self.dropout(F.relu(self.function1(x))))

class resconnect_and_layernorm(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(resconnect_and_layernorm, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        # res first and then norm
        return self.norm(x + self.dropout(sublayer_output))

class Embedding(nn.Module):
    def __init__(self, vocab_size, input_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.input_dim = input_dim

    def forward(self, x):
        return self.embedding(x) * np.sqrt(self.input_dim)

class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, input_dim,device=device)
        pos = torch.arange(0, max_len,device=device).unsqueeze(1)
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (torch.arange(0, input_dim, 2,device=device) / input_dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (torch.arange(0, input_dim, 2,device=device) / input_dim)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output =  x + self.encoding.unsqueeze(0)[:, :x.shape[1], :]       
        output = self.dropout(output)
        return output

class Encoder(nn.Module):
    def __init__(self, n_heads, input_dim, N_layer, hidden_dim, dropout=0.1):
        super(Encoder, self).__init__()
        self.self_attention = nn.ModuleList(multihead_attn(n_heads, input_dim) for _ in range(N_layer))
        self.res_layer1 = nn.ModuleList([resconnect_and_layernorm(input_dim, dropout) for _ in range(N_layer)])
        self.res_layer2 = nn.ModuleList([resconnect_and_layernorm(input_dim, dropout) for _ in range(N_layer)])
        self.feedforward = nn.ModuleList([feedforwardLayer(input_dim, input_dim, hidden_dim, dropout) for _ in range(N_layer)])
        self.norm = nn.LayerNorm(input_dim)
        self.N_layer = N_layer

    def forward(self, x, mask):
        for i in range(self.N_layer):
            x = self.res_layer1[i](x, self.self_attention[i](x, x, x, mask))
            x = self.res_layer2[i](x, self.feedforward[i](x))
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, n_heads, input_dim, N_layer, hidden_dim, dropout=0.1):
        super(Decoder, self).__init__()
        self.self_attention = nn.ModuleList(multihead_attn(n_heads, input_dim) for _ in range(N_layer))
        self.cross_attention = nn.ModuleList(multihead_attn(n_heads, input_dim) for _ in range(N_layer))
        self.feedforward = nn.ModuleList([feedforwardLayer(input_dim, input_dim, hidden_dim, dropout) for _ in range(N_layer)])
        self.res_layer1 = nn.ModuleList([resconnect_and_layernorm(input_dim, dropout) for _ in range(N_layer)])
        self.res_layer2 = nn.ModuleList([resconnect_and_layernorm(input_dim, dropout) for _ in range(N_layer)])
        self.res_layer3 = nn.ModuleList([resconnect_and_layernorm(input_dim, dropout) for _ in range(N_layer)])
        self.N_layer = N_layer


    def forward(self, x, memory, src_mask, tgt_mask):
        for i in range(self.N_layer):
            x = self.res_layer1[i](x, self.self_attention[i](x, x, x, tgt_mask))
            x = self.res_layer2[i](x, self.cross_attention[i](x, memory, memory, src_mask))
            x = self.res_layer3[i](x, self.feedforward[i](x))
        return x
        
class TransformerModule(nn.Module):
    def __init__(self):
        super(TransformerModule, self).__init__()
        self.src_embed = nn.Sequential(Embedding(len(vocab), input_dim), PositionalEncoding(input_dim))
        self.tgt_embed = nn.Sequential(Embedding(len(vocab), input_dim), PositionalEncoding(input_dim))
        self.encoder = Encoder(n_heads,input_dim,N_layer,hidden_dim)
        self.decoder = Decoder(n_heads,input_dim,N_layer,hidden_dim)
        self.Linear = nn.Linear(input_dim, len(vocab))
    
    def forward(self,x, y,x_mask, y_mask):
        x = self.src_embed(x)
        memory = self.encoder(x, x_mask)
        y = self.tgt_embed(y)
        output = self.decoder(x, memory, x_mask, y_mask)
        return F.softmax(self.Linear(output),dim=-1)
        

def train_step(net, src, tgt, loss, optimizer):
    optimizer.zero_grad()
    src_mask , tgt_mask = create_mask(src, tgt[:,:-1])
    output = net(src, tgt[:,:-1],src_mask, tgt_mask)
    output = output.reshape(-1, output.size(-1)) 
    labels = tgt[:,1:].reshape(-1)  
    loss_v = loss(output, labels)/14
    loss_v.backward()
    optimizer.step()
    return loss_v

def val_step(net, src, tgt, loss):
    src_mask , tgt_mask = create_mask(src, tgt[:,:-1])
    output = net(src, tgt[:,:-1],src_mask, tgt_mask)
    tgt_onehot = torch.nn.functional.one_hot(tgt[:,1:], num_classes=len(vocab))
    tgt_onehot = tgt_onehot.float()
    loss_v = loss(output, tgt_onehot)
    return loss_v

def accuracy():
    for src, tgt in dl_train:
        break
    src_mask, tgt_mask = create_mask(src, tgt[:,:-1])
    loss_fn = F.cross_entropy
    preds = net(src, tgt[:,:-1],src_mask, tgt_mask)
    preds = preds.reshape(-1, preds.size(-1))
    labels = tgt[:,1:].reshape(-1)
    loss = loss_fn(preds, labels)
    print(loss.item())
    preds = preds.argmax(dim=-1).view(-1)[labels!=0]  
    labels = labels[labels!=0]  
    acc = (preds==labels).sum()/(labels==labels).sum()  
    print('acc=',acc.item())  
    print(preds)

def train_model(net, dl_train, dl_val, loss, optimizer, epochs):
    for epoch in range(epochs):
        net.train()
        i = 0
        for src, tgt in dl_train:
            i += 1
            loss_v = train_step(net, src, tgt, loss, optimizer)
            print("epoch", epoch,"batchs", i, "loss", loss_v.item())
        net.eval()
        for src, tgt in dl_val:
            loss_val_v = val_step(net, src, tgt, loss)
            loss_val_sum += loss_val_v.item()
        print("epoch", epoch)
        accuracy()

# loss is labelsmotthingloss
class LabelSmoothingLoss(nn.Module):  
    "Implement label smoothing."  
    def __init__(self, size, padding_idx, smoothing=0.0): #size为词典大小  
        super(LabelSmoothingLoss, self).__init__()  
        self.criterion = nn.KLDivLoss(reduction="sum")  
        self.padding_idx = padding_idx  
        self.confidence = 1.0 - smoothing  
        self.smoothing = smoothing  
        self.size = size  
        self.true_dist = None  
          
    def forward(self, x, target):  
        assert x.size(1) == self.size  
        true_dist = x.data.clone()  
        true_dist.fill_(self.smoothing / (self.size - 2))  #预测结果不会是<SOS> #和<PAD>  
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)  
        true_dist[:, self.padding_idx] = 0  
        mask = torch.nonzero((target.data == self.padding_idx).int())  
        if mask.dim() > 0:  
            true_dist.index_fill_(0, mask.squeeze(), 0.0)  
        self.true_dist = true_dist  
        print(x)
        print(true_dist)
        return self.criterion(x, true_dist) 

loss_fn = LabelSmoothingLoss(size=len(vocab),   
            padding_idx=0, smoothing=0.2) 
net = TransformerModule().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
train_model(net, dl_train, dl_val, loss_fn, optimizer, 100)
