import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义参数解析器
def parse_args():
    parser = argparse.ArgumentParser(description="使用微调的 GPT-2 模型进行摘要预测。")
    parser.add_argument('--input_file', type=str, default='./data/public_test.jsonl', help="输入 JSONL 文件的路径。")
    parser.add_argument('--output_file', type=str, default='submission.jsonl', help="保存输出 JSONL 文件的路径。")
    parser.add_argument('--model_path', type=str, default='./gpt2-finetuned-summarization', help="微调的 GPT-2 模型的路径。")
    parser.add_argument('--batch_size', type=int, default=4, help="推理时的批量大小。")
    parser.add_argument('--max_input_length', type=int, default=512, help="tokenization 的最大输入长度。")
    parser.add_argument('--max_target_length', type=int, default=64, help="生成的最大目标长度。")
    parser.add_argument('--num_beams', type=int, default=4, help="Beam search 的 beam 数量。")
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = 'left'  # 设置左填充

    # 添加特殊标记
    special_tokens = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>', 'sep_token': '<SEP>'}
    tokenizer.add_special_tokens(special_tokens)

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.resize_token_embeddings(len(tokenizer))
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
        input_ids_list = []
        attention_masks_list = []

        for inp in inputs:
            # Tokenize maintext
            tokenized_input = tokenizer(
                inp,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_input_length - 2  # 留出空间给特殊标记
            )

            # 构建 input_ids
            input_ids = (
                [tokenizer.convert_tokens_to_ids('<BOS>')] +
                tokenized_input['input_ids'] +
                [tokenizer.convert_tokens_to_ids('<SEP>')]
            )

            # 构建 attention_mask
            attention_mask = [1] * len(input_ids)

            # 填充或截断序列
            padding_length = args.max_input_length - len(input_ids)
            if padding_length > 0:
                # 左填充
                input_ids = [tokenizer.pad_token_id] * padding_length + input_ids
                attention_mask = [0] * padding_length + attention_mask
            else:
                # 截断
                input_ids = input_ids[-args.max_input_length:]
                attention_mask = attention_mask[-args.max_input_length:]

            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_mask)

        model_inputs = {
            'input_ids': input_ids_list,
            'attention_mask': attention_masks_list
        }

        return model_inputs

    # 对测试数据进行预处理
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

    # 设置格式为 PyTorch tensors
    tokenized_test_dataset.set_format(type='torch')

    # 创建数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest', return_tensors='pt')

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
                max_new_tokens=args.max_target_length,
                num_beams=args.num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids('<EOS>')
            )

            # 解码生成的摘要
            for gen_ids in generated_ids:
                gen_ids = gen_ids.tolist()

                # 在左填充的情况下，直接处理生成的序列
                # 在 <EOS> 处截断
                eos_id = tokenizer.convert_tokens_to_ids('<EOS>')
                if eos_id in gen_ids:
                    eos_index = gen_ids.index(eos_id)
                    gen_ids = gen_ids[:eos_index]

                # 去除输入部分（因为生成的序列可能包含输入）
                # 找到 <SEP> 的位置
                sep_id = tokenizer.convert_tokens_to_ids('<SEP>')
                if sep_id in gen_ids:
                    sep_index = gen_ids.index(sep_id) + 1  # 包含 <SEP>
                    gen_ids = gen_ids[sep_index:]
                else:
                    gen_ids = []

                pred = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                predictions.append(pred.strip())

    # 将预测结果和对应的 id 配对，保存为 submission.jsonl
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for id_, title in zip(test_ids, predictions):
            output = {"title": title, "id": id_}
            json_line = json.dumps(output, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"预测结果已保存至 {args.output_file}")

if __name__ == "__main__":
    main()
