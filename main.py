import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from argparse import ArgumentParser
import os
import numpy as np
import random
import json

from tokenizer import SimpleTokenizer
from dataset import CLSDataset, LMDataset
from utilities import Utilities

from model import Encoder, Decoder, EncoderForSpeechCLS
from transformers import BertTokenizer


seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 16
block_size = 32
learning_rate = 1e-3
n_embd = 64
n_head = 2
n_layer = 4
dropout = 0.1

eval_interval = 100
max_iters = 500
eval_iters = 200

n_input = 64
n_hidden = 100
n_output = 3
epochs_CLS = 15


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_texts(file_dir):
    texts = []
    for file_name in os.listdir(file_dir):
        if 'test' in file_name:
            continue
        texts.append(open(os.path.join(file_dir, file_name), 'r', encoding='utf-8').read())
    return texts


def collate_batch(batch):
    data, labels = zip(*batch)
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]
    padded_sequences = F.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    classifier.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        
        accuracy = (100 * total_correct / total_samples)
        return accuracy


def compute_perplexity(decoderLM, data_loader, eval_iters=100):
    decoderLM.eval()
    losses = []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        _, _, loss = decoderLM(X, Y)
        losses.append(loss.item())
        if len(losses) >= eval_iters:
            break
    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()

    decoderLM.train()
    return perplexity


def train_CLS(tokenizer, embedding_method, pos_emb_method='abs', is_moe=False):
    train_CLS_dataset = CLSDataset(tokenizer, 'speechesdataset/train_CLS.tsv')
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True, num_workers=0)

    test_CLS_dataset = CLSDataset(tokenizer, 'speechesdataset/test_CLS.tsv')
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False, num_workers=0)

    cls_model = EncoderForSpeechCLS(
        label_num=3,
        vocab_size=tokenizer.vocab_size,
        d_model=n_embd,
        hidden_size=n_hidden,
        block_size=block_size,
        head_num=n_head,
        layer_num=n_layer,
        dropout_rate=dropout,
        method=embedding_method,
        pos_emb_method=pos_emb_method,
        is_moe=is_moe,
    )

    cls_model.to(device)
    optimizer = torch.optim.AdamW(cls_model.parameters(), lr=learning_rate)

    global_step = 0
    running_loss = 0.0

    print(f"Start Training CLS using {device} device")
    if is_moe:
        epochs_CLS = 35
    else:
        epochs_CLS = 15
    for epoch in range(epochs_CLS):
        running_loss = 0.0
        count = 0
        if epoch == 0:
            acc = compute_classifier_accuracy(cls_model, test_CLS_loader)
            print(f"epoch: {epoch:^2}, global step: {global_step:>4}, running_loss: {running_loss:.4f}, accuracy: {acc:.4f}")
        epoch += 1

        for xb, yb in train_CLS_loader:
            global_step += 1
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = cls_model(input_ids=xb, labels=yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1

        acc = compute_classifier_accuracy(cls_model, test_CLS_loader)
        running_loss /= count
        print(f"epoch: {epoch:^2}, global step: {global_step:>4}, running_loss: {running_loss:.4f}, accuracy: {acc:.4f}")


def train_LM(tokenizer, pos_emb_method='abs', is_moe=False):
    train_LM_dataset = LMDataset(tokenizer, 'speechesdataset/train_LM.txt', block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)

    test_loader_dict = {}
    for file in os.listdir('./speechesdataset'):
        if 'test_LM' in file:
            name = file.split('_')[-1].split('.')[0]
            test_LM_dataset = LMDataset(tokenizer, os.path.join('./speechesdataset', file), block_size)
            test_LM_loader = DataLoader(test_LM_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            test_loader_dict[name] = test_LM_loader

    lm_model = Decoder(
        vocab_size=tokenizer.vocab_size,
        d_model=n_embd,
        hidden_size=256,
        block_size=block_size,
        head_num=n_head,
        layer_num=n_layer,
        dropout_rate=dropout,
        pos_emb_method=pos_emb_method,
        is_moe=is_moe,
    )
    lm_model.to(device)
    optimizer = torch.optim.AdamW(lm_model.parameters(), lr=learning_rate)

    running_loss = 0.0
    count = 0
    if is_moe:
        max_iters = 1000
    else:
        max_iters = 500
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        _, _, loss = lm_model(input_ids=xb, label_ids=yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        
        running_loss += loss.item()
        count += 1

        if i % 100 == 99:
            running_loss /= count
            ppl_dict = {'Train': round(compute_perplexity(lm_model, train_LM_loader, eval_iters=i), 4)}
            for name in test_loader_dict:
                loader = test_loader_dict[name]
                ppl = compute_perplexity(lm_model, loader, eval_iters=1e10)
                ppl_dict[name] = round(ppl, 4)
            print(f"Step: {i+1:>3}, running_loss: {running_loss:.4f}, Perplexity: {ppl_dict}")
            running_loss = 0.0


def main(args):
    set_seed(seed)
    print('Loading data and creating tokenizer ...')
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts))
    print("Vocabulary size: ", tokenizer.vocab_size)

    print('Sanity Check ...')
    if args.task == 'part1':
        model = Encoder(vocab_size=tokenizer.vocab_size, d_model=n_embd, hidden_size=n_hidden, block_size=block_size, head_num=n_head, layer_num=n_layer)
    elif args.task == 'part2':
        model = Decoder(vocab_size=tokenizer.vocab_size, d_model=n_embd, hidden_size=n_hidden, block_size=block_size, head_num=n_head, layer_num=n_layer)
    else:
        pass
    
    if args.task != 'part3':
        utilities = Utilities(tokenizer, model)
        utilities.sanity_check(texts[0].split('\n')[0], block_size)

    if args.task == 'part1':
        for embedding_method in ['avg', 'cls']:
            print(f"\nTraining CLS - embedding: {embedding_method}...")
            train_CLS(tokenizer, embedding_method)
    elif args.task == 'part2':
        print("\nTraining LM ...")
        train_LM(tokenizer)
    elif args.task == 'part3':
        tokenizer_dict = {'simple': tokenizer, 'pretrained': BertTokenizer.from_pretrained('bert-base-uncased')}
        print('\n\n\n----part 3.1----')
        for name in tokenizer_dict:
            print(f"\n==== {name} tokenizer ====")
            tknz = tokenizer_dict[name]

            print(f"\n\t CLS Part")
            for emb in ['cls', 'avg']:
                print(f"using [{name}] tokenizer, [{emb}] embedding, [sin] position embedding")
                train_CLS(tknz, embedding_method='cls', pos_emb_method='sin')
            

            print(f"\n\t LM Part")
            for pos_emb in ['abs', 'sin']:
                print(f"using [{name}] tokenizer, [{pos_emb}] position embedding")
                train_LM(tknz, pos_emb_method=pos_emb, is_moe=False)
                
        print(f"\n\n\n----part 3.2----")
            
        print(f"====CLS==== using [pretrained] tokenizer, [avg] embedding, [sin] position embedding, MoE model")
        train_CLS(tokenizer_dict['pretrained'], embedding_method='avg', pos_emb_method='sin', is_moe=True)

        print(f"====LM====using [pretrained] tokenizer, [abs] position embedding, MoE model")
        train_LM(tokenizer_dict['pretrained'], pos_emb_method='abs', is_moe=True)

    else:
        raise ValueError("Invalid task argument. Use 'cls' or 'lm'.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default='part1', choices=['part1', 'part2', 'part3'])
    args = parser.parse_args()

    main(args)
