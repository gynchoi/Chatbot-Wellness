import torch
from kobert_transformers import get_tokenizer
from torch.utils.data import Dataset


class WellnessClassificationDataset(Dataset):
    def __init__(self,
               file_path = "drive/MyDrive/Colab Notebooks/dataset/wellness_classification.txt",
               num_label = 359,
               device = 'cpu',
               max_seq_len = 512, 
               tokenizer = get_tokenizer()
               ):
        self.file_path = file_path
        self.device = device
        self.data =[]
        self.tokenizer = tokenizer

        f = open(self.file_path, 'r', encoding='utf-8')

        while True:
            line = f.readline()
            if not line:
                break
            dataset = line.split("    ")
            
            input_ids = self.tokenizer.encode(dataset[0])
            token_type_ids = [0] * len(input_ids)
            attention_mask = [1] * len(input_ids)
            
            padding_size = max_seq_len - len(input_ids)
            input_ids += [0] * padding_size
            token_type_ids += [0] * padding_size
            attention_mask += [0] * padding_size
            
            label = int(dataset[1][:-1])

            data = {
                    'input_ids': torch.tensor(input_ids).to(self.device),
                    'token_type_ids': torch.tensor(token_type_ids).to(self.device),
                    'attention_mask': torch.tensor(attention_mask).to(self.device),
                    'labels': torch.tensor(label).to(self.device)
                    }

            self.data.append(data)

        f.close()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        item = self.data[index]
        return item

