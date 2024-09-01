import os
import vessl
import numpy as np
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
from transformers import GPT2Config, AdamW
from kogpt2_transformers import get_kogpt2_tokenizer, get_kogpt2_model

class WellnessAutoRegressiveDataset(Dataset):
  def __init__(self,
               path = "/input/wellness_autoregressive.txt",
               n_ctx = 1024
               ):
    self.path = path
    self.data =[]
    self.tokenizer = get_kogpt2_tokenizer()

    bos = [self.tokenizer.bos_token_id]
    eos = [self.tokenizer.eos_token_id]
    pad = [self.tokenizer.pad_token_id]

    f = open(self.path, 'r', encoding='utf-8')

    while True:
      line = f.readline()
      if not line: 
        break
      datas = line.split("    ")
      index_of_words = bos + self.tokenizer.encode(datas[0]) + eos + bos + self.tokenizer.encode(datas[1][:-1])+ eos
      pad_token_len = n_ctx - len(index_of_words)
      index_of_words += pad * pad_token_len
      self.data.append(index_of_words)

    f.close()

  def __len__(self):
    return len(self.data)

  def __getitem__(self,index):
    item = self.data[index]
    return item


def get_kogpt2_config():
  kogpt2_config = {
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "n_ctx": 1024,
        "n_embd": 768,
        "n_head": 12,
        "n_layer": 12,
        "n_positions": 1024,
        "vocab_size": 50000,
        "activation_function": "gelu"
    }
  
  return GPT2Config.from_dict(kogpt2_config)

class KoGPT2Dialogue(nn.Module):
  def __init__(self):
    super(KoGPT2Dialogue, self).__init__()
    self.kogpt2 = get_kogpt2_model()

  def generate(self,
               input_ids,
               do_sample=True,
               max_length= 60,
               top_p=0.92,
               top_k=50,
               temperature= 0.6,
               no_repeat_ngram_size =None,
               num_return_sequences=3,
               early_stopping=False,
               ):
      
    return self.kogpt2.generate(input_ids,
               do_sample=do_sample,
               max_length=max_length,
               top_p = top_p,
               top_k=top_k,
               temperature=temperature,
               no_repeat_ngram_size= no_repeat_ngram_size,
               num_return_sequences=num_return_sequences,
               early_stopping = early_stopping,
              )

  def forward(self, input, labels = None):
    if labels is not None:
      outputs = self.kogpt2(input, labels=labels)
    else:
      outputs = self.kogpt2(input)

    return outputs


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='PyTorch Dialogue')
  parser.add_argument('--input-path', type=str, default='/input', help='input dataset path')
  parser.add_argument('--output-path', type=str, default='/output', help='output files path')
  parser.add_argument('--checkpoint-path', type=str, default='/output/checkpoint', help='checkpoint path')
  args = parser.parse_args()

 
  if not os.path.exists(args.checkpoint_path):
    print(f"Make Checkpoint Path: {args.checkpoint_path}")
    os.makedirs(args.checkpoint_path)
  else:
    print("Checkpoint Path already exists")
    
  save_path = os.path.join(args.checkpoint_path, "kogpt-wellness-autoregressive.pth")
  print(f"Save model in: {save_path}")
  
  # hyperparameters
  lr = float(os.environ.get('lr', 5e-5))
  epochs = int(os.environ.get('epochs', 5))
  batch_size = int(os.environ.get('batch_size', 8))
  print("Hyperparameters: lr(" + str(lr) + ") epochs(" + str(epochs) + ") batch_size(" + str(batch_size) + ")")
  save_step = 100
  
  # Load Data from input
  dataset= WellnessAutoRegressiveDataset(path=args.input_path + '/wellness_autoregressive.txt')
  train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
  
  # Validate device
  ctx = "cuda" if torch.cuda.is_available() else "cpu"
  device = torch.device(ctx)
  print(f'Device: {device}')
  print(f'Device count: {torch.cuda.device_count()}')

  model = KoGPT2Dialogue()
  model.to(device)

  criterion = torch.nn.CrossEntropyLoss(ignore_index=3)
  # Prepare optimizer and schedule (linear warmup and decay)
  no_decay = ['bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  
  print("Start Training") 
  losses =[]
  for epoch in range(epochs):
    count = 0
    with tqdm(total=len(train_loader), desc=f"Train({epoch})") as pbar:
      for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = torch.stack(data)  
        data = data.transpose(1, 0)
        data= data.to(ctx)

        outputs = model(data, labels=data)
        _, logits = outputs[:2]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = data[..., 1:].contiguous()

        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # if count % 10 == 0:
        #     print('epoch no.{} train no.{}  loss = {}'.format(epoch, count + 1, loss))
        if (count > 0 and count % save_step == 0) or (len(data) < batch_size):
          torch.save({
            'epoch': epoch,
            'train_no': count,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
          }, save_path)
        count += 1
        pbar.update(1)
        pbar.set_postfix_str(f"Loss: {loss.item():.3f} Avg Loss: ({np.mean(losses):.3f})")
        vessl.log(step=epoch+1, payload={'loss': loss.item()})
