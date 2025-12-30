from sacrebleu.metrics import BLEU
from data import tokenizer,model
from tqdm.auto import tqdm
# 测试循环

from sacrebleu.metrics import BLEU
import numpy as np
bleu = BLEU() #目标序列是英文，所以无需指定分词器


#训练循环
from tqdm.auto import tqdm

def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        batch_data = batch_data.to(device)
        outputs = model(**batch_data)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward() # 计算梯度
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

    
def test_loop(dataloader, model):
    preds, labels = [], []
    
    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=max_length,
            ).cpu().numpy() #获得模型生成的token ID并转换为NumPy数组
        label_tokens = batch_data["labels"].cpu().numpy() #将标签也转到CPU并转换为NumPy数组
        
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        #这里我们将标签序列中的 -100 替换为 pad token ID 以便于分词器解码
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [pred.strip() for pred in decoded_preds]
        labels += [[label.strip()] for label in decoded_labels]

    return bleu.corpus_score(preds, labels).score