## Improving Distantly-Supervised Named Entity Recognition with Self-Collaborative Denoising Learning (EMNLP 2021 Accept-Findings)

## Framework
![image](img/fw.png)

## Requirements

- python==3.7.4
- pytorch==1.6.0
- [huggingface transformers](https://github.com/huggingface/transformers)
- numpy
- tqdm

## Overview

```
├── root
│   └── dataset
│       ├── conll03_train.json
│       ├── conll03_dev.json
│       ├── conll03_test.json
│       ├── conll03_tag_to_id.json
│       └── ...
│   └── models
│       ├── __init__.py
│       └── modeling_roberta.py
│   └── utils
│       ├── __init__.py
│       ├── config.py
│       ├── data_utils.py
│       ├── eval.py
│       └── ...
│   └── ptms
│       └── ... (trained results, e.g., saved models, log file)
│   └── cached_models
│       └── ... (RoBERTa pretrained model, which will be downloaded automatically)
│   └── run_script.py
│   └── run_script.sh
```

## How to run
```console
sh run_script.sh <GPU ID> <DATASET NAME>
```
e.g., 
```console
sh run_script.sh 0 conll03
```
Specific parameters for different datasets can be found in our paper, and then modify them in ```run_script.sh```.

## Citation
```
@inproceedings{zhang-etal-2021,
    title = "Improving Distantly-Supervised Named Entity Recognition with Self-Collaborative Denoising Learning",
    author = "Zhang, Xinghua  and
      Yu, Bowen  and
      Liu, Tingwen  and
      Zhang, Zhenyu  and
      Sheng, Jiawei  and
      Xue, Mengge  and
      Xu, Hongbo",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Online and in the Barceló Bávaro Convention Centre, Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
}
```