# PyTorch NLP Template with Hydra & WandB ð¥

PyTorch template for easy use! (actually, for my ownð)
- This template especially focuses on **BERT-based NLP tasks**, but it can also be customized to any tasks.
- This template supports **distributed-data-parallel(ddp)** and **automatic-mixed-precision(amp)** training.
- This template includes a simple BERT classification code.
- This template includes simple [WandB](https://wandb.ai/site) and [Hydra](https://hydra.cc/) application.
- This template follows [black](https://github.com/psf/black) code formatting.

## 1. Structure
```sh
root/
ââ docker/
â  ââ Dockerfile
â  ââ generate_container.sh
â  ââ generate_image.sh
ââ scripts/
â  ââ run.sh
ââ src/
â  ââ base/
â  â  ââ base_trainer.py
â  ââ checkpoints/ # gitignored
â  â  ââ WANDB_RUN_ID/ # automatically generated
â  â    ââ best/
â  â    ââ latest/
â  ââ conf/
â  â  ââ defaults/
â  â    ââ cf_distributed.py
â  â    ââ cf_wandb.py
â  â  ââ cf_train.py
â  ââ loader.py
â  ââ main.py
â  ââ sweep.yaml
â  ââ trainer.py
â  ââ utils.py
ââ .gitignore
ââ LICENSE
ââ README.md
ââ requirements.txt
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
