import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def forward(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xmean = self.running_var
        
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
    
    def forward(self, x):
        xmean = x.mean(-1, keepdim=True)
        xvar = x.var(-1, keepdim=True)

        out = (x - xmean) / torch.sqrt(xvar + self.eps)
        out = self.gamma * out + self.beta

        return out


class Head(nn.Module):
    def __init__(self, d_model, head_size, block_size, dropout_rate=0.1, is_causal=True):
        super().__init__()
        self.key = nn.Linear(d_model, head_size)
        self.query = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)
        self.droput = nn.Dropout(dropout_rate)
        self.is_causal = is_causal
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = k @ q.transpose(-2, -1) * C ** -0.5  # (B, T, T)
        if self.is_causal:
            wei = wei.masked_fill(self.tril == 0, float('-inf'))
        attn_map = F.softmax(wei, dim=-1)
        wei = self.droput(attn_map)

        out = wei @ v
        return out, attn_map


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, block_size, head_num, droput_rate=0.1, is_causal=True):
        super().__init__()
        self.heads = nn.ModuleList([Head(d_model, d_model // head_num, block_size, droput_rate, is_causal) for _ in range(head_num)])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(droput_rate)

    def forward(self, x):
        out_list = [head(x) for head in self.heads]
        attn_maps = [out[1] for out in out_list]

        out = torch.cat([out[0] for out in out_list], dim=-1)
        out = self.dropout(self.proj(out))

        return out, attn_maps
    

class FFN(nn.Module):
    def __init__(self, d_model, hidden_size, dropout_rate=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_size)
        self.linear2 = nn.Linear(hidden_size, d_model)
        self.activation = nn.ReLU()
        self.droput = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x) 
        x = self.linear2(x)
        x = self.droput(x)

        return x


class Block(nn.Module):
    def __init__(self, d_model, hidden_size, block_size, head_num, dropout_rate=0.1, is_causal=True):
        super().__init__()
        self.sa = MultiHeadAttention(d_model, block_size, head_num, dropout_rate, is_causal)
        self.ffn = FFN(d_model, hidden_size, dropout_rate)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
    
    def forward(self, x):
        # x = x + self.sa(self.ln1(x))
        out, attn_maps = self.sa(self.ln1(x))
        x = x + out
        x = x + self.ffn(self.ln2(x))

        return x, attn_maps


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, hidden_size, block_size, head_num, layer_num, dropout_rate=0.1):
        super().__init__()
        self.is_causal = False
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, hidden_size, block_size, head_num, dropout_rate, is_causal=self.is_causal) for _ in range(layer_num)])

        self.ln_f = LayerNorm(d_model)

    
    def forward(self, input_ids, label_ids=None):
        batch_size, seq_len = input_ids.shape

        tok_emb = self.tok_embedding(input_ids)
        pos_emb = self.pos_embedding(torch.arange(seq_len))
        x = tok_emb + pos_emb

        attn_maps = []
        for block in self.blocks:
            x, attn_map = block(x)
            attn_maps.extend(attn_map)
        
        return x, attn_maps


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, hidden_size, block_size, head_num, layer_num, dropout_rate=0.1):
        super().__init__()
        self.is_causal = True
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, hidden_size, block_size, head_num, dropout_rate, is_causal=self.is_causal) for _ in range(layer_num)])

        self.ln_f = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    
    def forward(self, input_ids, label_ids=None):
        batch_size, seq_len = input_ids.shape

        tok_emb = self.tok_embedding(input_ids)
        pos_emb = self.pos_embedding(torch.arange(seq_len))
        x = tok_emb + pos_emb
        
        attn_maps = []
        for block in self.blocks:
            x, attn_map = block(x)
            attn_maps.extend(attn_map)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        batch_size, seq_len, vocab_size = logits.shape

        if label_ids is None:
            loss = None
        else:
            logits = logits.view(batch_size*seq_len, vocab_size)
            labels = label_ids.view(batch_size*seq_len)
            loss = F.cross_entropy(logits, labels)
        
        return logits, attn_maps, loss


class EncoderForSpeechCLS(nn.Module):
    def __init__(self, label_num, vocab_size, d_model, hidden_size, block_size, head_num, layer_num, dropout_rate, method='avg'):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, hidden_size, block_size, head_num, layer_num, dropout_rate)
        self.linear = nn.Linear(d_model, label_num)
        self.method = method
    
    def forward(self, input_ids, labels=None):
        outputs, _ = self.encoder(input_ids)
        if self.method == 'avg':
            emb = outputs.mean(dim=1)
        elif self.method == 'cls':
            emb = outputs[:, 0, :]
        else:
            raise ValueError(f"SpeechCLS method should be cls or avg, but recieved: {self.method}")

        logits = self.linear(emb)

        if labels is None:
            return logits
        else:
            loss = F.cross_entropy(logits, labels)
            return logits, loss
