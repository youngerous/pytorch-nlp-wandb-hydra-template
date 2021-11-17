# PyTorch NLP Template🔥

PyTorch template for easy use! (actually, for my own😉)
- This template especially focuses on **BERT-based NLP tasks**, but it can also be customized to any tasks.
- This template supports **distributed-data-parallel(ddp)** and **automatic-mixed-precision(amp)** training.
- This template includes a simple BERT classification code.
- This template follows [black](https://github.com/psf/black) code formatting.

## 1. Structure
```sh
scripts/
  └─ run.sh
src/
  base/
    ├─ __init__.py
    └─ base_trainer.py
  ├─ __init__.py
  ├─ config.py
  ├─ loader.py
  ├─ main.py
  ├─ trainer.py
  └─ utils.py
├─ .gitignore
├─ Dockerfile
├─ LICENSE
├─ README.md
└─ requirements.txt
```

## 2. Requirements
- torch==1.7.1
- transformers==4.9.1
- datasets==1.11.0

More dependencies are written in [requirements.txt](https://github.com/youngerous/pytorch-nlp-template/blob/main/requirements.txt).

## 3. Usage

### 3.1. Set docker environments
```sh
# Example: generate image 
$ docker build -t $image_name --build-arg UNAME=$(whoami) --build-arg UID=$(id -u) --build-arg GID=$(id -g) .

# Example: generate container with two gpus
$ docker run --gpus '"device=0,1"' -td --ipc=host --name $container_name -v ~/repo:/repo -v /hdd/data/:/hdd/data/ -v /etc/passwd:/etc/passwd -p 8888:8888 -p 6006:6006 $image_name

# Example: start container
$ docker exec -it $container_name bash
```

### 3.2. Set container environments
```sh
$ pip install -r requirements.txt
```

### 3.3. Train model
```sh
$ sh scripts/run.sh
```

## 4. Sample Experiment Results

|           Task           | Dataset | Model | Test Accuracy |
| :----------------------: | :-----: | :---: | :-----------: |
| Sentiment Classification |  IMDB   | BERT  |      93%      |

## 5. LICENSE
[MIT License](https://github.com/youngerous/pytorch-nlp-template/blob/main/LICENSE)


## 6. TODO
- [ ] setuptools
- [ ] max_step option
- [ ] saving several checkpoints (e.g. saving top-5 ckpts)