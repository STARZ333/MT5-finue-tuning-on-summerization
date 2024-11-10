import json
import torch
import argparse
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义参数解析器
def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for summarization using fine-tuned MT5 model.")
    parser.add_argument('--input_file', type=str, default='./data/public_test.jsonl', help="Path to the input JSONL file.")
    parser.add_argument('--output_file', type=str, default='submission.jsonl', help="Path to save the output JSONL file.")
    parser.add_argument('--model_path', type=str, default='./ADLHW2/mt5_beam_search_num_beams_8', help="Path to the fine-tuned MT5 model.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for inference.")
    parser.add_argument('--max_input_length', type=int, default=512, help="Maximum input length for tokenization.")
    parser.add_argument('--max_target_length', type=int, default=64, help="Maximum target length for generation.")
    parser.add_argument('--num_beams', type=int, default=4, help="Number of beams for beam search.")
    return parser.parse_args()

# 加载测试数据
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    # 解析参数
    args = parse_args()

    # 加载保存的模型和 tokenizer
    tokenizer = MT5Tokenizer.from_pretrained(args.model_path)
    model = MT5ForConditionalGeneration.from_pretrained(args.model_path)
    model = model.to(device)

    # 加载测试数据
    test_data = load_jsonl(args.input_file)

    # 创建 HuggingFace Dataset
    test_dataset = Dataset.from_list(test_data)

    # 记录原始的 id 列表
    test_ids = test_dataset['id']

    # 数据预处理函数
    def preprocess_function(examples):
        inputs = examples['maintext']
        model_inputs = tokenizer(inputs, max_length=args.max_input_length, truncation=True)
        return model_inputs

    # 对测试数据进行预处理
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

    # 设置格式为 PyTorch tensors
    tokenized_test_dataset.set_format(type='torch')

    # 创建数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')

    # 创建 DataLoader
    test_dataloader = DataLoader(
        tokenized_test_dataset, batch_size=args.batch_size, collate_fn=data_collator
    )

    # 开始生成摘要
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 生成摘要
            generated_ids = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=args.max_target_length,
                num_beams=args.num_beams,
                early_stopping=True
            )

            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend(preds)

    # 将预测结果和对应的 id 配对，保存为 submission.jsonl
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for id_, title in zip(test_ids, predictions):
            output = {"title": title.strip(), "id": id_}
            json_line = json.dumps(output, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"预测结果已保存至 {args.output_file}")

if __name__ == "__main__":
    main()
