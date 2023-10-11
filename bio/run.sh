# Parameter settings for full-shot scenarios.

## Infomax

### GPF

python prompt_tuning_full_shot.py --input_model_file pretrained_models/infomax.pth --tuning_type gpf

### GPF-plus

python prompt_tuning_full_shot.py --input_model_file pretrained_models/infomax.pth --tuning_type gpf-plus --pnum 20 

## EdgePred

### GPF

python prompt_tuning_full_shot.py --input_model_file pretrained_models/edgepred.pth --tuning_type gpf

### GPF-plus

python prompt_tuning_full_shot.py --input_model_file pretrained_models/edgepred.pth --tuning_type gpf-plus --pnum 5

## AttrMasking

### GPF

python prompt_tuning_full_shot.py --input_model_file pretrained_models/masking.pth --tuning_type gpf

### GPF-plus

python prompt_tuning_full_shot.py --input_model_file pretrained_models/masking.pth --tuning_type gpf-plus --pnum 5

## ContextPred

### GPF

python prompt_tuning_full_shot.py --input_model_file pretrained_models/contextpred.pth --tuning_type gpf

### GPF-plus

python prompt_tuning_full_shot.py --input_model_file pretrained_models/contextpred.pth --tuning_type gpf-plus --pnum 5

## GCL

### GPF

python prompt_tuning_full_shot.py --input_model_file pretrained_models/gcl.pth --lr 1e-4 --dropout_ratio 0 --tuning_type gpf

### GPF-plus

python prompt_tuning_full_shot.py --input_model_file pretrained_models/gcl.pth --lr 1e-4 --dropout_ratio 0 --tuning_type gpf-plus --num_layers 2 --pnum 5