# MT5-finue-tuning-on-summerization
Dataset & evaluation script for ADL 2024 homework 2


## How to start?
1. Create conda environment.
  ```bash
  conda create --name [your_env_name] python=3.8
  conda activate [your_env_name]
  ```
2. Install library
  ```
  pip install -r requirements.txt
  git clone https://github.com/deankuo/ADL24-HW2.git
  cd ADL24-HW2
  pip install -e tw_rouge
  ```

## Data Prepare
You can access and download my fine-tuned model and dataset using the provided
[download link](https://drive.google.com/drive/folders/1a1Yk_gkwvMp8YsllYqwOcF7GXk0w33XC?usp=sharing)  .

## Train
To fine-tune the ```google/mt5-small``` model on the summarization task, you can run the following command:
```bash
python train.py
```

## Predict
To perform predictions on the test data using the fine-tuned model, you can run the following command:

```bash
python infer.py --input_file a.jsonl --output_file b.jsonl
```
P.S. Ensure that your dataset follows the same format as the provided data.

## Evaluation
To evaluate finetuned model on public test dataset, you can use the Script
```
usage: eval.py [-h] [-r REFERENCE] [-s SUBMISSION]

optional arguments:
  -h, --help            show this help message and exit
  -r REFERENCE, --reference REFERENCE
  -s SUBMISSION, --submission SUBMISSION
```

Example:
```
python eval.py -r public.jsonl -s submission.jsonl
{
  "rouge-1": {
    "f": 0.21999419163162043,
    "p": 0.2446195813913345,
    "r": 0.2137398792982201
  },
  "rouge-2": {
    "f": 0.0847583291303246,
    "p": 0.09419044877345074,
    "r": 0.08287844474014894
  },
  "rouge-l": {
    "f": 0.21017939117006337,
    "p": 0.25157090570020846,
    "r": 0.19404349000921203
  }
}
```
Or use Python Library
```
>>> from tw_rouge import get_rouge
>>> get_rouge('我是人', '我是一個人')
{'rouge-1': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}, 'rouge-2': {'f': 0.33333332888888895, 'p': 0.5, 'r': 0.25}, 'rouge-l': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}}
>>> get_rouge(['我是人'], [ '我是一個人'])
{'rouge-1': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}, 'rouge-2': {'f': 0.33333332888888895, 'p': 0.5, 'r': 0.25}, 'rouge-l': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}}
>>> get_rouge(['我是人'], ['我是一個人'], avg=False)
[{'rouge-1': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}, 'rouge-2': {'f': 0.33333332888888895, 'p': 0.5, 'r': 0.25}, 'rouge-l': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}}]
```


### Results
Beam search with num_beams=8 as the final generation strategy.

<table>
  <tr>
    <td>Model</td>
    <td>rouge-1</td>
    <td>rouge-2</td>
    <td>rouge-L</td>
  </tr>

  <tr>
    <td>mt5-small</td>
    <td>23.8</td>
    <td>9.5</td>
    <td>21.5</td>
  </tr>
  <tr>
    <td>baseline</td>
    <td>22.0</td>
    <td>8.5</td>
    <td>20.5</td>
  </tr>
<table>

## Reference
[cccntu/tw_rouge](https://github.com/cccntu/tw_rouge)
