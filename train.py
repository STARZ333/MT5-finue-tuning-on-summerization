import json
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import nltk
nltk.download('punkt')

# 新增导入 matplotlib 用于绘图
import matplotlib.pyplot as plt

def load_jsonl(path):
  data = []
  with open(path, 'r', encoding='utf-8') as f:
      for line in f:
          data.append(json.loads(line))
  return data

# 加载数据
train_data = load_jsonl('data/train.jsonl')
val_data = load_jsonl('data/public.jsonl')

from datasets import Dataset

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')

max_input_length = 512
max_target_length = 64

def preprocess_function(examples):
  inputs = examples['maintext']
  targets = examples['title']
  model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
  
  # 设置标签
  labels = tokenizer(targets, max_length=max_target_length, truncation=True)
  
  model_inputs['labels'] = labels['input_ids']
  return model_inputs

# 预处理数据
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

# 设置格式为 PyTorch tensors
tokenized_train_dataset.set_format(type='torch')
tokenized_val_dataset.set_format(type='torch')

# 加载模型
model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 创建数据整理器
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# 创建 DataLoader
batch_size = 8  # 根据显存大小调整

train_dataloader = DataLoader(
  tokenized_train_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator
)
val_dataloader = DataLoader(
  tokenized_val_dataset, batch_size=batch_size, collate_fn=data_collator
)

# 设置优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 8
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

# 训练循环
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 新增：用于记录训练信息的列表
train_losses = []
val_rouge1 = []
val_rouge2 = []
val_rougeL = []

for epoch in range(epochs):
  model.train()
  epoch_loss = 0
  progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}')
  for batch in progress_bar:
      batch = {k: v.to(device) for k, v in batch.items()}

      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()

      epoch_loss += loss.item()

      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()

      progress_bar.set_postfix({'loss': loss.item()})

  avg_loss = epoch_loss / len(train_dataloader)
  train_losses.append(avg_loss)  # 记录平均损失
  print(f"Epoch {epoch+1} average loss: {avg_loss}")

  # 在验证集上评估
  model.eval()
  predictions = []
  references = []
  with torch.no_grad():
      for batch in tqdm(val_dataloader, desc='Evaluating'):
          batch = {k: v.to(device) for k, v in batch.items()}

          generated_ids = model.generate(
              input_ids=batch['input_ids'],
              attention_mask=batch['attention_mask'],
              max_length=max_target_length,
              num_beams=4,
              early_stopping=True
          )
          preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
          
          # 处理 labels 中的 -100
          labels = batch['labels']
          labels = labels.clone()
          labels[labels == -100] = tokenizer.pad_token_id
          refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

          predictions.extend(preds)
          references.extend(refs)

  # 计算 ROUGE 分数
  from tw_rouge import get_rouge

  rouge_scores = get_rouge(predictions, references)
  print(f"Validation ROUGE scores: {rouge_scores}")

  # 记录验证集的 ROUGE 分数
  val_rouge1.append(rouge_scores['rouge-1']['f'])
  val_rouge2.append(rouge_scores['rouge-2']['f'])
  val_rougeL.append(rouge_scores['rouge-l']['f'])

  # 保存每个 epoch 的训练信息
  with open('training_log.txt', 'a', encoding='utf-8') as f:
      f.write(f"Epoch {epoch+1}:\n")
      f.write(f"  Average Training Loss: {avg_loss}\n")
      f.write(f"  Validation ROUGE Scores:\n")
      f.write(f"    ROUGE-1 F1 Score: {rouge_scores['rouge-1']['f']}\n")
      f.write(f"    ROUGE-2 F1 Score: {rouge_scores['rouge-2']['f']}\n")
      f.write(f"    ROUGE-L F1 Score: {rouge_scores['rouge-l']['f']}\n")
      f.write("\n")

# 保存模型
model.save_pretrained('./mt5-finetuned-summarization')
tokenizer.save_pretrained('./mt5-finetuned-summarization')

# 绘制学习曲线并保存
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(12, 6))
plt.plot(epochs_range, val_rouge1, label='ROUGE-1 F1 Score')
plt.plot(epochs_range, val_rouge2, label='ROUGE-2 F1 Score')
plt.plot(epochs_range, val_rougeL, label='ROUGE-L F1 Score')
plt.xlabel('Epoch')
plt.ylabel('ROUGE F1 Score')
plt.title('Validation ROUGE Scores over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('rouge_scores.png')  # 保存图像为文件
