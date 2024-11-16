from datasets import load_dataset

# Load only the test split from the dataset
test_data = load_dataset("cfilt/iitb-english-hindi", split="test")

# You now have the test data
print(test_data)





import torch
#from datasets import load_dataset
from tokenizers import tokenizers
from tokenizers.models import WordLevel
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
import random
random.seed(42)
def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

config = get_config()

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


# Build tokenizers
tokenizer_src = get_or_build_tokenizer(config, test_data, 'en')
tokenizer_tgt = get_or_build_tokenizer(config, test_data, 'hi')



import torch
import torch.nn as nn
from torch.utils.data import Dataset

class EnToHinDataset(Dataset):

    def __init__(self, ds, tk_src, tk_tgt, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tk_src = tk_src
        self.tk_tgt = tk_tgt

        # Special tokens
        self.sos = torch.tensor([tk_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos = torch.tensor([tk_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad = torch.tensor([tk_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Get the source and target text from the dataset
        dic = self.ds[idx]
        src_text = dic['translation']['en']
        tgt_text = dic['translation']['hi']

        # Tokenize the source and target text
        en = self.tk_src.encode(src_text).ids
        de = self.tk_tgt.encode(tgt_text).ids

        # Calculate the number of padding tokens needed
        enc_pad_len = self.seq_len - len(en) - 2  # for <sos> and <eos>
        dec_pad_len = self.seq_len - len(de) - 1  # only <sos> at the beginning

        # Check if the sentence is too long
        if enc_pad_len < 0 or dec_pad_len < 0:
            raise ValueError("Sentence is too long")

        # Create the encoder input by adding <sos>, <eos>, and padding
        en_inp = torch.cat([
            self.sos,
            torch.tensor(en, dtype=torch.int64),
            self.eos,
            torch.tensor([self.pad] * enc_pad_len, dtype=torch.int64)
        ])

        # Create the decoder input by adding <sos> and padding
        de_inp = torch.cat([
            self.sos,
            torch.tensor(de, dtype=torch.int64),
            torch.tensor([self.pad] * dec_pad_len, dtype=torch.int64)
        ])

        # Create the label by adding <eos> at the end and padding
        label = torch.cat([
            torch.tensor(de, dtype=torch.int64),
            self.eos,
            torch.tensor([self.pad] * dec_pad_len, dtype=torch.int64)
        ])

        # Return a dictionary containing the inputs and labels
        return {
            "encoder_input": en_inp,  # Encoder input
            "decoder_input": de_inp,  # Decoder input
            "label": label,  # Target labels
            "encoder_mask": (en_inp != self.pad).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (de_inp != self.pad).unsqueeze(0).int() & causal_mask(de_inp.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

causal_mask(3)



from torch.utils.data import Dataset, DataLoader, random_split
# Train and test splits
data = test_data
train_ds_size = int(0.9 * len(data))
val_ds_size = len(data) - train_ds_size
train_data, val_data = random_split(data, [train_ds_size, val_ds_size])




max_len_src = 0
max_len_tgt = 0
for item in test_data:
  src_ids = tokenizer_src.encode(item['translation']['en']).ids
  tgt_ids = tokenizer_tgt.encode(item['translation']['hi']).ids
  max_len_src = max(max_len_src, len(src_ids))
  max_len_tgt = max(max_len_tgt, len(tgt_ids))
print(f'Max length of source sentence: {max_len_src}')
print(f'Max length of target sentence: {max_len_tgt}')




seq_len = 100

t_ds = EnToHinDataset(train_data, tokenizer_src, tokenizer_tgt, 100)
v_ds = EnToHinDataset(val_data, tokenizer_src, tokenizer_tgt, 100)


from torch.utils.data import DataLoader

t_dl = DataLoader(t_ds, batch_size=16, shuffle=True)
v_dl = DataLoader(v_ds, batch_size=16, shuffle=True)


for i in t_dl:
  print(i['encoder_input'].shape)
  print(i['decoder_input'].shape)
  print(i['label'].shape)
  print(i['encoder_mask'].shape)
  print(i['decoder_mask'].shape)
  break




src_vocab_size = tokenizer_src.get_vocab_size()
tgt_vocab_size = tokenizer_tgt.get_vocab_size()






import torch
import torch.nn as nn
from torch.nn import functional as F

vocab_size = 100
batch_size = 16
block_size = 100  # seq_len
n_embd = 50
n_head = 2
n_layer = 2
dropout = 0.2
torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 10000 # max no of steps for training 
eval_interval = 100 # after how many steps the evaluation will take place
eval_iters = 1000  # how many sample of batches will use for evaluation

lr = 3e-4




class DecoderHead(nn.Module):
    """ one head self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.tri = torch.tril(torch.ones(block_size, block_size))
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', self.tri)


    def forward(self, x, y ,z, mask):

        mask = mask.squeeze(1)
        B, T, C = x.shape
        k = self.key(x) # (B, T, h)
        q = self.query(x) # (B, T, h)
        v = self.value(x) # (B, T, h)
        
      

        att = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,h)@(B,h,T) -> (B,T,T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim= -1) # (B,T,T)
        att = self.dropout(att)

        out = att @ v # (B,T,T)@(B,T,h) -> (B,T,h)
        return out

class EnocderHead(nn.Module):
    """ one head self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.tri = torch.tril(torch.ones(block_size, block_size))
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', self.tri)


    def forward(self, x, mask=None):

        mask = mask.reshape(16,1, 100)
        B, T, C = x.shape
        k = self.key(x) # (B, T, h)
        q = self.query(x) # (B, T, h)
        v = self.value(x) # (B, T, h)

        att = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,h)@(B,h,T) -> (B,T,T)
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim= -1) # (B,T,T)
        att = self.dropout(att)

        out = att @ v # (B,T,T)@(B,T,h) -> (B,T,h)

        return out

class EncoderMultiHeadAtt(nn.Module):

    def __init__(self,num_heads,  head_size):
        super().__init__()
        self.hd = nn.ModuleList([EnocderHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x, mask):
        out = torch.cat([h(x, mask) for h in self.hd], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class DecoderMultiHeadAtt(nn.Module):

    def __init__(self,num_heads,  head_size):
        super().__init__()
        self.hd = nn.ModuleList([DecoderHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y ,z, mask):
        out = torch.cat([h(x,  y ,z, mask) for h in self.hd], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
class EncoderBlock(nn.Module):

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.mhead = EncoderMultiHeadAtt(n_head, head_size )
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, s_mask):

        # x = x + self.mhead(x)   #skip connections
        # x = x + self.ffwd(x)

        x = x + self.mhead(self.ln1(x), s_mask)
        x = x + self.ffwd(self.ln2(x))
        return x
class DecoderBlock(nn.Module):

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.mhead = DecoderMultiHeadAtt(n_head, head_size)
        self.croshead = DecoderMultiHeadAtt(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, e_out, s_mask, t_mask):

        # x = x + self.mhead(x)   #skip connections
        # x = x + self.ffwd(x)
        x = self.ln1(x)
        x = x + self.mhead(x, e_out, e_out, s_mask)
        x = x + self.mhead(self.ln1(x), e_out, e_out, s_mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )


    def forward(self, x):
        return self.net(x)

class GPTFromScratch(nn.Module):

    def __init__(self):
        super().__init__()
        self.embd_table_e = nn.Embedding(src_vocab_size, n_embd) # (vocab_size,C)
        self.pos_table_e = nn.Embedding(block_size, n_embd) # (T,C)

        self.embd_table_d = nn.Embedding(tgt_vocab_size, n_embd) # (vocab_size,C)
        self.pos_table_d = nn.Embedding(block_size, n_embd) # (T,C)


        # self.encoderblock = nn.Sequential(
        #                 EncoderBlock(n_embd, n_head),
        #                 EncoderBlock(n_embd, n_head),
        #                 EncoderBlock(n_embd, n_head),
        #                 EncoderBlock(n_embd, n_head),
        #                 )
        #self.encoderblock = EncoderBlock(n_embd, n_head)

        self.encoderblock = nn.ModuleList([EncoderBlock(n_embd, n_head) for _ in range(4)])
                     
        self.decoderblock = nn.ModuleList([DecoderBlock(n_embd, n_head) for _ in range(4)])
        # nn.Sequential(
        #                 DecoderBlock(n_embd, n_head),
        #                 DecoderBlock(n_embd, n_head),
        #                 DecoderBlock(n_embd, n_head),
        #                 DecoderBlock(n_embd, n_head),
        #                 )

        #self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])


        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        #self.lm_head_e = nn.Linear(n_embd, src_vocab_size)
        self.lm_head_d = nn.Linear(n_embd, tgt_vocab_size)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def encoder(self, src_idx, mask):
        B, T = src_idx.shape  # (B -> batch, T -> block_size(seq_len))

        # both xb, yb shape is (B,T) tensor of ints
        tok_emb = self.embd_table_e(src_idx) #o/p -> (B,T,C)
        pos_emb = self.pos_table_e(torch.arange(T, device=device))
        x  = tok_emb + pos_emb # (B,T,C)-> (B,T,C)+ (C,T)
        #x = self.head(x)
        # x = self.mhead(x)
        # x = self.ffwd(x)
        #x = self.encoderblock(x, mask)
        for block in self.encoderblock:
            x = block(x, mask)
        x = self.ln_f(x) # (B,T,C)
        return x
        #logits = self.lm_head_e(x) # (B,T,vocab_size)


    def decoder(self, tgt_idx,  e_out, s_mask, t_mask):
        B, T = tgt_idx.shape  # (B -> batch, T -> block_size(seq_len))

        # both xb, yb shape is (B,T) tensor of ints
        tok_emb = self.embd_table_d(tgt_idx) #o/p -> (B,T,C)
        pos_emb = self.pos_table_d(torch.arange(T, device=device))
        x  = tok_emb + pos_emb # (B,T,C)-> (B,T,C)+ (C,T)
        #x = self.head(x)
        # x = self.mhead(x)
        # x = self.ffwd(x)
        #x = self.decoderblock(x, e_out, s_mask, t_mask)

        for block in self.decoderblock:
            x = block(x, e_out, s_mask, t_mask)
        x = self.ln_f(x) # (B,T,C)

        logits = self.lm_head_d(x) # (B,T,vocab_size)

        return logits

    def forward(self, idx, targets=None):
        B, T = idx.shape  # (B -> batch, T -> block_size(seq_len))

        # both xb, yb shape is (B,T) tensor of ints
        tok_emb = self.embd_table(idx) #o/p -> (B,T,C)
        pos_emb = self.pos_table(torch.arange(T, device=device))
        x  = tok_emb + pos_emb # (B,T,C)-> (B,T,C)+ (C,T)
        #x = self.head(x)
        # x = self.mhead(x)
        # x = self.ffwd(x)
        x = self.block(x)
        x = self.ln_f(x) # (B,T,C)

        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_token):
        # idx is (B.T) array
        for _ in range(max_new_token):
            #crop ids to only consider last block_size tokens
            idx_cond =  idx[:, -block_size:] #(B,T)
            # predictions
            logits, loss = self(idx_cond)  #(B,T,C)
            # take only last time step
            logits = logits[:, -1, :]  #(B,-1, C) -> (B, T+1th, C)
            probs = F.softmax(logits, dim = -1)  #(B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            # append sample in the running sequence
            idx = torch.cat((idx,idx_next), dim = 1) #(B, T+1)

        return idx  #(B, T+1)







model = GPTFromScratch()
m = model.to(device)
# print the no of params in the model
total_params = sum(p.numel() for p in m.parameters())
print(f"The total no of params in the model is {total_params}")

# create torch optimiser:
optimizer = torch.optim.AdamW(model.parameters(), lr= lr)


# training loop:
for iter in range(max_iters):

    # # evaluation of Loss on train and val
    # if iter % eval_interval == 0 or iter == max_iters -1:
    #     losses = estimate_loss()
    #     print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data

    s_xb =  i['encoder_input']
    t_xb =    i['decoder_input']
    s_msk= i['encoder_mask']
    t_msk= i['decoder_mask']



    en_out = model.encoder(s_xb,s_msk)

    de_out = model.decoder(t_xb,  en_out, s_msk, t_msk)

    print(en_out.shape)
    print(de_out.shape)
    break

    #print('loss:', loss.item())
    #print(logits.shape)
    # set grad = zero
    optimizer.zero_grad(set_to_none=True)

    # back propagation:
        # grad calculation and param update
    loss.backward()
    optimizer.step()