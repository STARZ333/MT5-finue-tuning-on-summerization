import json

# 定义输入和输出文件名
input_file = 'data/public.jsonl'
output_file = 'data/public_test.jsonl'

# 打开输入文件和输出文件
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # 解析每一行的JSON数据
        data = json.loads(line)
        
        # 创建新的格式，包含所需的字段
        new_format = {
            "date_publish": data["date_publish"],
            "source_domain": data["source_domain"],
            "maintext": data["maintext"],
            "split": data["split"],
            "id": data["id"]
        }
        
        # 将新的格式写入输出文件
        outfile.write(json.dumps(new_format, ensure_ascii=False) + '\n')

# 输出创建的文件名
print("Created/Modified files during execution:")
print(output_file)