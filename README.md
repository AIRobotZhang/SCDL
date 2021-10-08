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
### 1. Training
```console
sh run_script.sh <GPU ID> <DATASET NAME> <IF TRAINING=True> <IF TESTING=False>
```
e.g., 
```console
sh run_script.sh 0 conll03 True False
```

### 2. Testing
```console
sh run_script.sh <GPU ID> <DATASET NAME> <IF TRAINING=False> <IF TESTING=True>
```
e.g., 
```console
sh run_script.sh 0 conll03 False True
```

