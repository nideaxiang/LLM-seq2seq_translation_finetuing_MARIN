from data import TRANS,collote_fn,model
from transformers import AutoTokenizer
from torch.utils.data import random_split
from transformers import AutoModelForSeq2SeqLM
import torch
from torch.utils.data import DataLoader

from model import train_loop,test_loop
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')



#保护显卡人人有责
max_dataset_size = 10000
train_set_size = 8000
valid_set_size = 2000

data = TRANS('data/translation2019zh/translation2019zh_train.json')
# 随机划分训练集和验证集,避免捕捉到顺序上的信息
train_data, valid_data = random_split(data, [train_set_size, valid_set_size])
# 测试集就200
test_data = TRANS('data/translation2019zh/translation2019zh_valid.json',max_dataset_size=200)

model_checkpoint = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)



from transformers import  get_scheduler
from torch.optim import AdamW

learning_rate = 2e-5
epoch_num = 3

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_bleu = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss)
    valid_bleu = test_loop(valid_dataloader, model, mode='Valid')
    print(f"BLEU: {valid_bleu:>0.2f}\n")
    if valid_bleu > best_bleu:
        best_bleu = valid_bleu
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_bleu_{valid_bleu:0.2f}_model_weights.bin')
print("Done!")


import json

model.load_state_dict(torch.load('epoch_1_valid_bleu_53.38_model_weights.bin'))

model.eval()
with torch.no_grad():
    print('evaluating on test set...')
    sources, preds, labels = [], [], []
    for batch_data in tqdm(test_dataloader):
        batch_data = batch_data.to(device)
        generated_tokens = model.generate(
            batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            max_length=max_length,
        ).cpu().numpy()
        label_tokens = batch_data["labels"].cpu().numpy()

        decoded_sources = tokenizer.batch_decode(
            batch_data["input_ids"].cpu().numpy(), 
            skip_special_tokens=True, 
            use_source_tokenizer=True
        )
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        sources += [source.strip() for source in decoded_sources]
        preds += [pred.strip() for pred in decoded_preds]
        labels += [[label.strip()] for label in decoded_labels]
    bleu_score = bleu.corpus_score(preds, labels).score
    print(f"Test BLEU: {bleu_score:>0.2f}\n")
    results = []
    print('saving predicted results...')
    for source, pred, label in zip(sources, preds, labels):
        results.append({
            "sentence": source, 
            "prediction": pred, 
            "translation": label[0]
        })
    with open('test_data_pred.json', 'wt', encoding='utf-8') as f:
        for exapmle_result in results:
            f.write(json.dumps(exapmle_result, ensure_ascii=False) + '\n')
