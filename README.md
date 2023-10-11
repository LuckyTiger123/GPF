# GPF
This is a Pytorch code Implementation of the paper [*Universal Prompt Tuning for Graph Neural Networks*](https://arxiv.org/abs/2209.15240), which is accepted by the NeurIPS 2023. We provide two graph prompt methods **GPF** and **GPF-plus** to perform prompt tuning during the downstream adaptations.

## Installation

We used the following packages under `Python 3.7`.

```
pytorch 1.4.0
torch-cluster 1.5.2
torch-geometric 1.0.3
torch-scatter 2.0.3
torch-sparse 0.5.1
torch-spline-conv 1.2.0
rdkit 2022.3.4
tqdm 4.31.1
tensorboardX 1.6
```

## Pre-trained models

The pre-tiraned models we use follow the training steps of the paper [*Strategies for Pre-training Graph Neural Networks*](https://github.com/snap-stanford/pretrain-gnns) and [*Graph Contrastive Learning with Augmentations*](https://github.com/Shen-Lab/GraphCL). For each pre-training dataset, we provide five basic pre-trained models, and the brief descriptions of their training strategies are as follows:

- Deep Graph Infomax (denoted by **Infomax**)
  It obtains expressive representations for graphs or nodes via maximizing the mutual information between graph-level representations and substructure-level represen- tations of different granularity.
- Edge Prediction (denoted by **EdgePred**)
  It is a regular graph reconstruction task used by many models, such as GAE. The prediction target is the existence of edge between a pair of nodes.
- Attribute Masking (denoted by **AttrMasking**)
  It masks node/edge attributes and then let GNNs predict those attributes based on neighboring structure.
- Context Prediction (denoted by **ContextPred**)
  It uses subgraphs to predict their surrounding graph structures, and aims to mapping nodes appearing in similar structural contexts to nearby embeddings.
- Graph Contrastive Learning (denoted by **GCL**)
  It embeds augmented versions of the anchor close to each other (positive samples) and pushes the embeddings of other samples (negatives) apart. We use the augmentation strategies proposed in *Graph Contrastive Learning with Augmentations* for generating the positive and negative samples.

## Dataset

The pre-training and downstream datasets used in our experiments are referred to the paper *Strategies for Pre-training Graph Neural Networks*. You can download the biology and chemistry datasets from [their repository](https://github.com/snap-stanford/pretrain-gnns). 

To run the codes successfully, the downloaded datasets should be placed in `/dataset` under `bio/` and `chem/`.

## Downstream adaptation

### Biology dataset

For the biology dataset, please run `prompt_tuning_full_shot.py` and `prompt_tuning_few_shot.py` under `bio/` for downstream adaptations. 

For the normal full-shot scenarios:

```python
usage: prompt_tuning_full_shot.py [-h] [--device DEVICE]
                        [--epochs EPOCHS] [--lr LR] [--decay DECAY]
                        [--num_layer NUM_LAYER] [--emb_dim EMB_DIM]
                        [--dropout_ratio DROPOUT_RATIO]
                        [--model_file MODEL_FILE]
                        [--tuning_type TUNING_TYPE]
                        [--seed SEED] [--runseed RUNSEED]
                        [--num_layers NUM_LAYERS] [--pnum PNUM]


optional arguments:
--device        		Which gpu to use if any (default: 0)
--epochs        		Number of epochs to train (default: 50)
--lr                            Learning rate (default: 0.0001)
--decay         		Weight decay (default: 0)
--num_layer			Number of GNN message passing layers (default: 5).
--emb_dim     			Embedding dimensions (default: 300)
--dropout_ratio 		Dropout ratio (default: 0.5)
--model_file			File path to read the model (if there is any)
--tuning_type		        'gpf' for GPF and 'gpf-plus' for GPF-plus in the paper
--seed	         		Seed for splitting dataset.
--runseed		  	Seed for running experiments.
--num_layers			A range of [1,2,3]-layer MLPs with equal width
--pnum	         		The number of independent basis for GPF-plus
```

For the few-show scenarios:

```python
usage: prompt_tuning_few_shot.py [-h] [--device DEVICE]
                  [--epochs EPOCHS] [--lr LR] [--decay DECAY]
                  [--num_layer NUM_LAYER] [--emb_dim EMB_DIM]
                  [--dropout_ratio DROPOUT_RATIO]
                  [--model_file MODEL_FILE]
                  [--tuning_type TUNING_TYPE] [--seed SEED]
                  [--runseed RUNSEED]
                  [--num_layers NUM_LAYERS] [--pnum PNUM]
                  [--shot_number SHOT_NUMBER]


optional arguments:
--device       			Which gpu to use if any (default: 0)
--epochs       			Number of epochs to train (default: 50)
--lr               	        Learning rate (default: 0.001)
--decay         		Weight decay (default: 0)
--num_layer			Number of GNN message passing layers (default: 5).
--emb_dim	  		Embedding dimensions (default: 300)
--dropout_ratio			Dropout ratio (default: 0.5)
--model_file 			File path to read the model (if there is any)
--tuning_type			'gpf' for GPF and 'gpf-plus' for GPF-plus in the paper
--seed         			Seed for splitting dataset.
--runseed	  		Seed for running experiments.
--num_layers			A range of [1,2,3]-layer MLPs with equal width
--pnum         			The number of independent basis for GPF-plus
--shot_number			Number of shots
```

### Chemistry dataset

For the chemistry dataset, please run `prompt_tuning_few_shot.py` and `prompt_tuning_full_shot.py` under `chem/` for downstream adaptations. 

For the normal full-shot scenarios:

```python
usage: prompt_tuning_full_shot.py [-h] [--device DEVICE]
                        [--epochs EPOCHS] [--lr LR] [--lr_scale LR_SCALE]
                        [--decay DECAY] [--num_layer NUM_LAYER]
                        [--emb_dim EMB_DIM] [--dropout_ratio DROPOUT_RATIO]
                        [--tuning_type TUNING_TYPE]
                        [--dataset DATASET] [--model_file MODEL_FILE]
                        [--seed SEED] [--runseed RUNSEED]
                        [--num_layers NUM_LAYERS] [--pnum PNUM]


optional arguments:
--device        		Which gpu to use if any (default: 0)
--epochs        		Number of epochs to train (default: 100)
--lr               	        Learning rate (default: 0.001)
--lr_scale    			Relative learning rate for the feature extraction layer (default: 1)
--decay          		Weight decay (default: 0)
--num_layer 			Number of GNN message passing layers (default: 5).
--emb_dim      			Embedding dimensions (default: 300)
--dropout_ratio  		Dropout ratio (default: 0.5)
--tuning_type 			'gpf' for GPF and 'gpf-plus' for GPF-plus in the paper
--dataset      			Root directory of dataset. For now, only classification.
--model_file 			File path to read the model (if there is any)
--seed            	        Seed for splitting the dataset.
--runseed      			Seed for minibatch selection, random initialization.
--split          		The way of dataset split(e.g., 'scaffold' for chem data)
--num_layers 			A range of [1,2,3]-layer MLPs with equal width
--pnum            	        The number of independent basis for GPF-plus
```

For the few-show scenarios:

```python
usage: prompt_tuning_few_shot.py [-h] [--device DEVICE]
                  [--epochs EPOCHS] [--lr LR] [--lr_scale LR_SCALE]
                  [--decay DECAY] [--num_layer NUM_LAYER] [--emb_dim EMB_DIM]
                  [--dropout_ratio DROPOUT_RATIO]
                  [--tuning_type TUNING_TYPE]
                  [--dataset DATASET] [--model_file MODEL_FILE] [--seed SEED]
                  [--runseed RUNSEED]
                  [--num_layers NUM_LAYERS] [--pnum PNUM]
                  [--shot_number SHOT_NUMBER]


optional arguments:
--device        		Which gpu to use if any (default: 0)
--epochs        		Number of epochs to train (default: 100)
--lr                            Learning rate (default: 0.001)
--lr_scale    			Relative learning rate for the feature extraction layer (default: 1)
--decay          		Weight decay (default: 0)
--num_layer  			Number of GNN message passing layers (default: 5).
--emb_dim      			Embedding dimensions (default: 300)
--dropout_ratio 		Dropout ratio (default: 0.5)
--tuning_type 			'gpf' for GPF and 'gpf-plus' for GPF-plus in the paper
--dataset      			Root directory of dataset. For now, only classification.
--model_file    		File path to read the model (if there is any)
--seed            	        Seed for splitting the dataset.
--runseed      			Seed for minibatch selection, random initialization.
--split          		The way of dataset split(e.g., 'scaffold' for chem data)
--num_layers 			A range of [1,2,3]-layer MLPs with equal width
--pnum            	        The number of independent basis for GPF-plus
--shot_number 			Number of shots
```

## Parameter settings

We have provided scripts with hyper-parameter settings to reproduce the experimental results presented in our paper. 

For the full-shot scenarios, you can obtain the experimental results by running `run.sh`.

```shell
sh run.sh
```

For the few-shot scenarios, you can obtain the experimental results by running `run_few_shot.sh`.

```shell
sh run_few_shot.sh
```

## Citation

You can cite our paper by following bibtex.

```tex
@inproceedings{Fang2023UniversalPT,
  title={Universal Prompt Tuning for Graph Neural Networks},
  author={Taoran Fang and Yunchao Zhang and Yang Yang and Chunping Wang and Lei Chen},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

