from torch.utils.data import Dataset, random_split
import json
import torch
from transformers import AutoModelForSeq2SeqLM

model_checkpoint = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)

#保护显卡人人有责
max_dataset_size = 10000
train_set_size = 8000
valid_set_size = 2000

class TRANS(Dataset):
    def __init__(self, data_file,max_dataset_size=10000):
        self.data = self.load_data(data_file,max_dataset_size)
    
    def load_data(self, data_file,max_dataset_size):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_dataset_size:
                    break
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['chinese'])
        batch_targets.append(sample['english'])
    batch_data = tokenizer(
        batch_inputs, 
        text_target=batch_targets, #定义目标序列的语言类型，分词器可以并行地处理源/目标语言
        padding=True, 
        max_length=max_length,
        truncation=True, 
        return_tensors="pt"
    )
    # 为 decoder 准备目标序列的 ids
    batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(batch_data['labels'])
    end_token_index = torch.where(batch_data['labels'] == tokenizer.eos_token_id)[1]
    #padding操作
    for idx, end_idx in enumerate(end_token_index):
        batch_data['labels'][idx][end_idx+1:] = -100
    return batch_data
