import random
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset,DataLoader  

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

word = '<PAD>,1,2,3,4,5,6,7,8,9,0,<SOS>,<EOS>,;'
vocab = {word: i for i, word in enumerate(word.split(','))} 
vocab_r = [k for k, v in vocab.items()] #反查词典  
device = 'cpu'

def get_data():
    num_to_be_sort = 10
    maxnum = 1000
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
class TwoSumDataset(torch.utils.data.Dataset):  
    def __init__(self,size = 100000):  
        super(Dataset, self).__init__()  
        self.size = size  
  
    def __len__(self):  
        return self.size  
  
    def __getitem__(self, i):  
        return get_data()  
      
ds_train = TwoSumDataset(size = 100000)  
ds_val = TwoSumDataset(size = 10000)  
  
  
# 数据加载器  
dl_train = DataLoader(dataset=ds_train,  
         batch_size=200,  
         drop_last=True,  
         shuffle=True)  
  
dl_val = DataLoader(dataset=ds_val,  
         batch_size=200,  
         drop_last=True,  
         shuffle=False)  
  
for src,tgt in dl_train:   
    break   

def create_mask(src, tgt):
    src_mask = (src != vocab['<PAD>']).unsqueeze(-2)
    tgt_mask = (tgt != vocab['<PAD>']).unsqueeze(-2)
    tgt_len = tgt.shape[-1]
    tgt_mask = tgt_mask & torch.tril(torch.ones(1,tgt_len, tgt_len,device=device)).bool()
    return src_mask, tgt_mask

class attn(nn.Module):
    def __init__(self):
        super(attn, self).__init__()

    def forward(self, Q, K, V,mask=None,dropout=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(K.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return p_attn.matmul(V), p_attn

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
        self.matrix = None

    
    def forward(self, Q, K, V, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = Q.shape[0]
        seq_len = Q.shape[1]
        input_dim = Q.shape[2]
        Q = self.WQ(Q)
        K = self.WK(K)
        V = self.WV(V)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.dim_each_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.dim_each_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.dim_each_head).transpose(1, 2)
        attn_score, self.matrix = self.attn(Q, K, V, mask,dropout=self.dropout)

        output = attn_score.transpose(1, 2).contiguous().view(batch_size, seq_len, input_dim)
        return self.WO(output)

class MaskedBatch():
    def __init__(self,src,tgt,pad=0):
        self.src = src
        self.tgt = tgt[:,:-1]
        self.tgt_y = tgt[:, 1:]
        self.src_mask,self.tgt_mask = create_mask(src,tgt[:,:-1])
        self.ntokens = (self.tgt_y!= pad).sum()

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
    def __init__(self, input_dim,vocab_size):
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
        self.norm = nn.LayerNorm(input_dim)


    def forward(self, x, memory, src_mask, tgt_mask):
        for i in range(self.N_layer):
            x = self.res_layer1[i](x, self.self_attention[i](x, x, x, tgt_mask))
            x = self.res_layer2[i](x, self.cross_attention[i](x, memory, memory, src_mask))
            x = self.res_layer3[i](x, self.feedforward[i](x))
        return self.norm(x)

class Generator(nn.Module):  
    "Define standard linear + softmax generation step."  
    def __init__(self, d_model, vocab):  
        super(Generator, self).__init__()  
        self.proj = nn.Linear(d_model, vocab)  
  
    def forward(self, x):  
        # return F.log_softmax(self.proj(x), dim=-1)  
        return self.proj(x)

class Transformer(nn.Module):  
    """  
    A standard Encoder-Decoder architecture. Base for this and many other models.  
    """  
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):  
        super(Transformer, self).__init__()  
        self.encoder = encoder  
        self.decoder = decoder  
        self.src_embed = src_embed  
        self.tgt_embed = tgt_embed  
        self.generator = generator  
        self.reset_parameters()  
          
    def forward(self, src, tgt, src_mask, tgt_mask):  
        "Take in and process masked src and target sequences."  
        return self.generator(self.decode(self.encode(src, src_mask),   
                src_mask, tgt, tgt_mask))  
      
    def encode(self, src, src_mask):  
        return self.encoder(self.src_embed(src), src_mask)  
      
    def decode(self, memory, src_mask, tgt, tgt_mask):  
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    @classmethod  
    def from_config(cls,src_vocab,tgt_vocab,N=6,d_model=512, d_ff=2048, h=8, dropout=0.1):  
        encoder = TransformerEncoder.from_config(N=N,d_model=d_model,  
                  d_ff=d_ff, h=h, dropout=dropout)  
        decoder = TransformerDecoder.from_config(N=N,d_model=d_model,  
                  d_ff=d_ff, h=h, dropout=dropout)  
        src_embed = nn.Sequential(WordEmbedding(d_model, src_vocab), PositionEncoding(d_model, dropout))  
        tgt_embed = nn.Sequential(WordEmbedding(d_model, tgt_vocab), PositionEncoding(d_model, dropout))  
          
        generator = Generator(d_model, tgt_vocab)  
        return cls(encoder, decoder, src_embed, tgt_embed, generator)  
      
    def reset_parameters(self):  
        for p in self.parameters():  
            if p.dim() > 1:  
                nn.init.xavier_uniform_(p)

class NoamOpt(torch.optim.AdamW):  
    def __init__(self, params, model_size=512, factor=1.0, warmup=4000,   
                 lr=0, betas=(0.9, 0.98), eps=1e-9,  
                 weight_decay=0, amsgrad=False):  
        super(NoamOpt,self).__init__(params, lr=lr, betas=betas, eps=eps,  
                 weight_decay=weight_decay, amsgrad=amsgrad)  
        self._step = 0  
        self.warmup = warmup  
        self.factor = factor  
        self.model_size = model_size  
          
    def step(self,closure=None):  
        "Update parameters and rate"  
        self._step += 1  
        rate = self.rate()  
        for p in self.param_groups:  
            p['lr'] = rate  
        super(NoamOpt,self).step(closure=closure)  
          
    def rate(self, step = None):  
        "Implement `lrate` above"  
        if step is None:  
            step = self._step  
        return self.factor * \
            (self.model_size ** (-0.5) *  
            min(step * self.warmup ** (-1.5),step ** (-0.5)))  

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
        return self.criterion(x, true_dist)

CrossLoss = nn.CrossEntropyLoss(reduction='sum')






#整体流程试算  
# test
  
for src,tgt in dl_train:  
    break   
mbatch = MaskedBatch(src=src,tgt=tgt,pad = 0)  
  
net = Transformer(Encoder(N_layer=3,input_dim=64, hidden_dim=128, n_heads=8, dropout=0.1),
                  Decoder(N_layer=3,input_dim=64, hidden_dim=128, n_heads=8, dropout=0.1),  
                    nn.Sequential(Embedding(input_dim=64, vocab_size = len(vocab)),   
                            PositionalEncoding(input_dim=64, dropout=0.1)),
                    nn.Sequential(Embedding(input_dim=64, vocab_size = len(vocab)),
                            PositionalEncoding(input_dim=64, dropout=0.1)),
                        Generator(d_model=64, vocab=len(vocab)))
  
#loss  
loss_fn = LabelSmoothingLoss(size=len(vocab),   
            padding_idx=0, smoothing=0.2) 
loss_fn = CrossLoss 
preds = net.forward(mbatch.src, mbatch.tgt, mbatch.src_mask, mbatch.tgt_mask)  
preds = preds.reshape(-1, preds.size(-1))  
labels = mbatch.tgt_y.reshape(-1)  
loss = loss_fn(preds, labels)/mbatch.ntokens   
print('loss=',loss.item())                               
  
#metric  
preds = preds.argmax(dim=-1).view(-1)[labels!=0]  
labels = labels[labels!=0]  
  
acc = (preds==labels).sum()/(labels==labels).sum()  
print('acc=',acc.item())  
  
  
from torchmetrics import Accuracy   
#使用torchmetrics中的指标  
accuracy = Accuracy(task='multiclass',num_classes=len(vocab))  
accuracy.update(preds,labels)  
print('acc=',accuracy.compute().item())  


# device = 'cuda'

# from torchkeras import KerasModel   
  
# class StepRunner:  
#     def __init__(self, net, loss_fn,   
#                  accelerator=None, stage = "train", metrics_dict = None,   
#                  optimizer = None, lr_scheduler = None  
#                  ):  
#         self.net,self.loss_fn,self.metrics_dict,self.stage = net,loss_fn,metrics_dict,stage  
#         self.optimizer,self.lr_scheduler = optimizer,lr_scheduler  
#         self.accelerator = accelerator  
#         if self.stage=='train':  
#             self.net.train()   
#         else:  
#             self.net.eval()  
      
#     def __call__(self, batch):  
#         src,tgt = batch   
#         mbatch = MaskedBatch(src=src,tgt=tgt,pad = 0)  
          
#         #loss  
#         with self.accelerator.autocast():  
#             preds = net.forward(mbatch.src, mbatch.tgt, mbatch.src_mask, mbatch.tgt_mask)  
#             preds = preds.reshape(-1, preds.size(-1))  
#             labels = mbatch.tgt_y.reshape(-1)  
#             loss = loss_fn(preds, labels)/mbatch.ntokens   
              
#             #filter padding  
#             preds = preds.argmax(dim=-1).view(-1)[labels!=0]  
#             labels = labels[labels!=0]  
  
  
#         #backward()  
#         if self.stage=="train" and self.optimizer is not None:  
#             self.accelerator.backward(loss)  
#             if self.accelerator.sync_gradients:  
#                 self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)  
#             self.optimizer.step()  
#             if self.lr_scheduler is not None:  
#                 self.lr_scheduler.step()  
#             self.optimizer.zero_grad()  
              
#         all_loss = self.accelerator.gather(loss).sum()  
#         all_preds = self.accelerator.gather(preds)  
#         all_labels = self.accelerator.gather(labels)  
          
          
#         #losses (or plain metrics that can be averaged)  
#         step_losses = {self.stage+"_loss":all_loss.item()}  
  
#         step_metrics = {self.stage+"_"+name:metric_fn(all_preds, all_labels).item()   
#                         for name,metric_fn in self.metrics_dict.items()}  
          
#         if self.stage=="train":  
#             if self.optimizer is not None:  
#                 step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']  
#             else:  
#                 step_metrics['lr'] = 0.0  
#         return step_losses,step_metrics  
      
# KerasModel.StepRunner = StepRunner   
  
# from torchmetrics import Accuracy   
  
# net = Transformer(Encoder(N_layer=5,input_dim=64, hidden_dim=128, n_heads=8, dropout=0.1),
#                   Decoder(N_layer=5,input_dim=64, hidden_dim=128, n_heads=8, dropout=0.1),  
#                     nn.Sequential(Embedding(input_dim=64, vocab_size = len(vocab)),   
#                             PositionalEncoding(input_dim=64, dropout=0.1)),
#                     nn.Sequential(Embedding(input_dim=64, vocab_size = len(vocab)),
#                             PositionalEncoding(input_dim=64, dropout=0.1)),
#                         Generator(d_model=64, vocab=len(vocab)))

# net.to(device)
  
# # loss_fn = LabelSmoothingLoss(size=len(vocab),   
# #             padding_idx=0, smoothing=0.1)  
# loss_fn = CrossLoss
  
# metrics_dict = {'acc':Accuracy(task='multiclass',num_classes=len(vocab))}   
# optimizer = NoamOpt(net.parameters(),model_size=64)  

# model1 = KerasModel(net,  
#                    loss_fn=loss_fn,  
#                    metrics_dict=metrics_dict,  
#                    optimizer = optimizer)  
  
# model1.fit(  
#     train_data=dl_train,  
#     val_data=dl_val,  
#     epochs=100,  
#     ckpt_path='checkpoint',  
#     patience=10,  
#     monitor='val_acc',  
#     mode='max',  
#     callbacks=None,  
#     plot=True  
# )  
