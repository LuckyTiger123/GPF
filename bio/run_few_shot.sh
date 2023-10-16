# Parameter settings for 50-shot scenarios.

## Infomax

### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/infomax.pth --tuning_type gpf

### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/infomax.pth --tuning_type gpf-plus --pnum 5 

## EdgePred

### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/edgepred.pth --tuning_type gpf

### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/edgepred.pth --tuning_type gpf-plus --pnum 20

## AttrMasking

### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/masking.pth --tuning_type gpf

### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/masking.pth --tuning_type gpf-plus --pnum 20

## ContextPred

### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/contextpred.pth --tuning_type gpf

### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/contextpred.pth --tuning_type gpf-plus --pnum 10

## GCL

### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/gcl.pth --tuning_type gpf --lr 1e-4 --dropout_ratio 0 --epochs 50 --num_layers 4

### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/gcl.pth --tuning_type gpf-plus --lr 1e-4 --dropout_ratio 0 --epochs 50 --num_layers 4 --pnum 20