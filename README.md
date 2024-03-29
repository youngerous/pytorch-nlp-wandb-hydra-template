# PyTorch NLP Template with Hydra & WandB 🔥

> Seems too complex to simply implement your research idea? Then how about [this template with pytorch-lightning](https://github.com/youngerous/pytorch-lightning-nlp-template)?

PyTorch template for easy use! (actually, for my own😉)
- This template especially focuses on **BERT-based NLP tasks**, but it can also be customized to any tasks.
- This template supports **distributed-data-parallel(ddp)** and **automatic-mixed-precision(amp)** training.
- This template includes a simple BERT classification code.
- This template includes simple [WandB](https://wandb.ai/site) and [Hydra](https://hydra.cc/) application.
- This template follows [black](https://github.com/psf/black) code formatting.

## 1. Structure
```sh
root/
├─ docker/
│  ├─ Dockerfile
│  ├─ generate_container.sh
│  └─ generate_image.sh
├─ scripts/
│  └─ run.sh
├─ src/
│  ├─ base/
│  │  └─ base_trainer.py
│  ├─ checkpoints/ # gitignored
│  │  └─ WANDB_RUN_ID/ # automatically generated
│  │    ├─ best/
│  │    └─ latest/
│  ├─ conf/
│  │  ├─ defaults/
│  │    ├─ cf_distributed.py
│  │    └─ cf_wandb.py
│  │  └─ cf_train.py
│  ├─ loader.py
│  ├─ main.py
│  ├─ sweep.yaml
│  ├─ trainer.py
│  └─ utils.py
├─ .gitignore
├─ LICENSE
├─ README.md
└─ requirements.txt
```

## 2. Requirements
- torch==1.7.1
- transformers==4.9.1
- datasets==1.11.0
- wandb==0.12.6
- hydra-core==1.1.1

More dependencies are written in [requirements.txt](https://github.com/youngerous/pytorch-nlp-wandb-hydra-template/blob/main/requirements.txt).

## 3. Usage

### 3.1. Set docker environments
```bash
# Example: generate image 
$ bash docker/generate_image.sh --image_name $IMAGE_NAME

# Example: generate container 
$ bash docker/generate_container.sh --image_name $IMAGE_NAME --container_name $CONTAINER_NAME --port_jupyter 8888 --port_tensorboard 6666

# Example: start container
$ docker exec -it $CONTAINER_NAME bash
```
### 3.2. Train model
```sh
$ sh scripts/run.sh
```

## 4. Sample Experiment Results

|           Task           | Dataset | Model | Test Accuracy |
| :----------------------: | :-----: | :---: | :-----------: |
| Sentiment Classification |  IMDB   | BERT  |      93%      |

## 5. LICENSE
[MIT License](https://github.com/youngerous/pytorch-nlp-wandb-hydra-template/blob/main/LICENSE)
