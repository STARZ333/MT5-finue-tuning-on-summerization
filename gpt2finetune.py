import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import nltk
nltk.download('punkt')

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

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# 使用 AutoTokenizer 和 AutoModelForCausalLM
model_name = 'uer/gpt2-chinese-cluecorpussmall'

# 设置 padding_side='left'
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

# 添加特殊标记
special_tokens = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>', 'sep_token': '<SEP>'}
tokenizer.add_special_tokens(special_tokens)

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

max_input_length = 512
max_target_length = 64

def preprocess_function(examples, is_train=True):
    inputs = examples['maintext']
    targets = examples['title']

    input_ids_list = []
    attention_masks_list = []
    labels_list = []

    for inp, tgt in zip(inputs, targets):
        # Tokenize maintext
        tokenized_input = tokenizer(
            inp,
            add_special_tokens=False,
            truncation=True,
            max_length=max_input_length - max_target_length - 3  # 留出空间给特殊标记和摘要
        )

        # Tokenize title (summary)
        tokenized_target = tokenizer(
            tgt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_target_length
        )

        if is_train:
            # 训练阶段：输入包括 maintext 和 title
            input_ids = (
                [tokenizer.convert_tokens_to_ids('<BOS>')] +
                tokenized_input['input_ids'] +
                [tokenizer.convert_tokens_to_ids('<SEP>')] +
                tokenized_target['input_ids'] +
                [tokenizer.convert_tokens_to_ids('<EOS>')]
            )

            # 构建 attention_mask
            attention_mask = [1] * len(input_ids)

            # 构建 labels
            labels = (
                [-100] * (len(tokenized_input['input_ids']) + 2) +  # +2 表示 <BOS> 和 <SEP>
                tokenized_target['input_ids'] +
                [tokenizer.convert_tokens_to_ids('<EOS>')]
            )
        else:
            # 验证阶段：输入只包括 maintext
            input_ids = (
                [tokenizer.convert_tokens_to_ids('<BOS>')] +
                tokenized_input['input_ids'] +
                [tokenizer.convert_tokens_to_ids('<SEP>')]  # 作为输入的结束标记
            )

            attention_mask = [1] * len(input_ids)

            # 在验证阶段，我们不需要 labels，但为了兼容性，仍然创建一个占位符
            labels = [-100] * len(input_ids)

        # 填充或截断序列
        padding_length = max_input_length - len(input_ids)
        if padding_length > 0:
            # 左填充
            input_ids = [tokenizer.pad_token_id] * padding_length + input_ids
            attention_mask = [0] * padding_length + attention_mask
            labels = [-100] * padding_length + labels
        else:
            # 截断
            input_ids = input_ids[-max_input_length:]
            attention_mask = attention_mask[-max_input_length:]
            labels = labels[-max_input_length:]

        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)
        labels_list.append(labels)

    model_inputs = {
        'input_ids': input_ids_list,
        'attention_mask': attention_masks_list,
        'labels': labels_list
    }

    return model_inputs

# 预处理训练数据
tokenized_train_dataset = train_dataset.map(
    lambda examples: preprocess_function(examples, is_train=True),
    batched=True,
    remove_columns=train_dataset.column_names
)

# 预处理验证数据
tokenized_val_dataset = val_dataset.map(
    lambda examples: preprocess_function(examples, is_train=False),
    batched=True,
    remove_columns=val_dataset.column_names
)


# 设置格式为 PyTorch tensors
tokenized_train_dataset.set_format(type='torch')
tokenized_val_dataset.set_format(type='torch')

# 数据整理器
from transformers import default_data_collator

data_collator = default_data_collator

# 创建 DataLoader
batch_size = 8  # 根据您的 GPU 显存进行调整

train_dataloader = DataLoader(
    tokenized_train_dataset,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator
)
val_dataloader = DataLoader(
    tokenized_val_dataset,
    batch_size=batch_size,
    collate_fn=data_collator
)

# 优化器和调度器
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
model = model.to(device)

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
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1} average loss: {avg_loss}")

    # 评估
    # 在评估阶段的循环中
    model.eval()
    predictions = []
    references = []
    sample_index = 0  # 初始化样本索引

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc='Evaluating'):
            batch = {k: v.to(device) for k, v in batch.items()}

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            # 生成摘要
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_target_length,
                num_beams=4,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids('<EOS>')
            )

            # 解码生成的摘要
            for i, gen_ids in enumerate(generated_ids):
                gen_ids = gen_ids.tolist()

                # 在左填充的情况下，不需要去除输入部分
                # 直接处理生成的序列

                # 在 <EOS> 处截断
                eos_id = tokenizer.convert_tokens_to_ids('<EOS>')
                if eos_id in gen_ids:
                    eos_index = gen_ids.index(eos_id)
                    gen_ids = gen_ids[:eos_index]

                pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
                predictions.append(pred)

                # 获取参考摘要
                ref = val_data[sample_index]['title']
                references.append(ref)

                sample_index += 1  # 更新样本索引

    # 计算 ROUGE 分数前，过滤空的预测和参考
    filtered_predictions = []
    filtered_references = []
    for pred, ref in zip(predictions, references):
        if pred.strip() and ref.strip():
            filtered_predictions.append(pred)
            filtered_references.append(ref)

    # 计算 ROUGE 分数
    from tw_rouge import get_rouge
    rouge_scores = get_rouge(filtered_predictions, filtered_references)
    print(f"Validation ROUGE scores: {rouge_scores}")



    # 计算 ROUGE 分数
    from tw_rouge import get_rouge

    rouge_scores = get_rouge(predictions, references)
    print(f"Validation ROUGE scores: {rouge_scores}")

    # 记录验证集的 ROUGE 分数
    val_rouge1.append(rouge_scores['rouge-1']['f'])
    val_rouge2.append(rouge_scores['rouge-2']['f'])
    val_rougeL.append(rouge_scores['rouge-l']['f'])

    # 保存训练信息
    with open('training_log.txt', 'a', encoding='utf-8') as f:
        f.write(f"Epoch {epoch+1}:\n")
        f.write(f"  Average Training Loss: {avg_loss}\n")
        f.write(f"  Validation ROUGE Scores:\n")
        f.write(f"    ROUGE-1 F1 Score: {rouge_scores['rouge-1']['f']}\n")
        f.write(f"    ROUGE-2 F1 Score: {rouge_scores['rouge-2']['f']}\n")
        f.write(f"    ROUGE-L F1 Score: {rouge_scores['rouge-l']['f']}\n")
        f.write("\n")

# 保存模型
model.save_pretrained('./gpt2-finetuned-summarization')
tokenizer.save_pretrained('./gpt2-finetuned-summarization')

# 绘制学习曲线
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
plt.savefig('rouge_scores.png')
