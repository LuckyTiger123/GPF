# Parameter settings for 50-shot scenarios.

#----------------------------------------------------------------------------------------------------------------------------------------

## BBBP

### Infomax

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/infomax.pth --dataset bbbp --tuning_type gpf --num_layers 2

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/infomax.pth --dataset bbbp --tuning_type gpf-plus --num_layers 3 --pnum 5

### EdgePred

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/edgepred.pth --dataset bbbp --tuning_type gpf --num_layers 2

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/edgepred.pth --dataset bbbp --tuning_type gpf-plus --num_layers 1 --pnum 10

### AttrMasking

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/masking.pth --dataset bbbp --tuning_type gpf --num_layers 1

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/masking.pth --dataset bbbp --tuning_type gpf-plus --num_layers 1 --pnum 10

### ContextPred

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/contextpred.pth --dataset bbbp --tuning_type gpf --num_layers 2

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/contextpred.pth --dataset bbbp --tuning_type gpf-plus --num_layers 2 --pnum 20

### GCL

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/gcl.pth --dataset bbbp --tuning_type gpf --num_layers 3

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/gcl.pth --dataset bbbp --tuning_type gpf-plus --num_layers 3 --pnum 10

#----------------------------------------------------------------------------------------------------------------------------------------

## Tox21

### Infomax

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/infomax.pth --dataset tox21 --tuning_type gpf --num_layers 3

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/infomax.pth --dataset tox21 --tuning_type gpf-plus --num_layers 2 --pnum 10

### EdgePred

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/edgepred.pth --dataset tox21 --tuning_type gpf --num_layers 2

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/edgepred.pth --dataset tox21 --tuning_type gpf-plus --num_layers 3 --pnum 5

### AttrMasking

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/masking.pth --dataset tox21 --tuning_type gpf --num_layers 3

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/masking.pth --dataset tox21 --tuning_type gpf-plus --num_layers 3 --pnum 5

### ContextPred

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/contextpred.pth --dataset tox21 --tuning_type gpf --num_layers 1

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/contextpred.pth --dataset tox21 --tuning_type gpf-plus --num_layers 3 --pnum 20

### GCL

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/gcl.pth --dataset tox21 --tuning_type gpf --num_layers 1

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/gcl.pth --dataset tox21 --tuning_type gpf-plus --num_layers 4 --pnum 10

#----------------------------------------------------------------------------------------------------------------------------------------

## ToxCast

### Infomax

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/infomax.pth --dataset toxcast --tuning_type gpf --num_layers 3

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/infomax.pth --dataset toxcast --tuning_type gpf-plus --num_layers 2 --pnum 10

### EdgePred

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/edgepred.pth --dataset toxcast --tuning_type gpf --num_layers 2

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/edgepred.pth --dataset toxcast --tuning_type gpf-plus --num_layers 3 --pnum 10

### AttrMasking

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/masking.pth --dataset toxcast --tuning_type gpf --num_layers 2

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/masking.pth --dataset toxcast --tuning_type gpf-plus --num_layers 2 --pnum 10

### ContextPred

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/contextpred.pth --dataset toxcast --tuning_type gpf --num_layers 2

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/contextpred.pth --dataset toxcast --tuning_type gpf-plus --num_layers 2 --pnum 5

### GCL

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/gcl.pth --dataset toxcast --tuning_type gpf --num_layers 4

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/gcl.pth --dataset toxcast --tuning_type gpf-plus --num_layers 2 --pnum 10

#----------------------------------------------------------------------------------------------------------------------------------------

## SIDER

### Infomax

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/infomax.pth --dataset sider --tuning_type gpf --num_layers 1

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/infomax.pth --dataset sider --tuning_type gpf-plus --num_layers 1 --pnum 5

### EdgePred

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/edgepred.pth --dataset sider --tuning_type gpf --num_layers 1

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/edgepred.pth --dataset sider --tuning_type gpf-plus --num_layers 2 --pnum 5

### AttrMasking

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/masking.pth --dataset sider --tuning_type gpf --num_layers 2

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/masking.pth --dataset sider --tuning_type gpf-plus --num_layers 1 --pnum 20

### ContextPred

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/contextpred.pth --dataset sider --tuning_type gpf --num_layers 1

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/contextpred.pth --dataset sider --tuning_type gpf-plus --num_layers 1 --pnum 5

### GCL

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/gcl.pth --dataset sider --tuning_type gpf --num_layers 4

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/gcl.pth --dataset sider --tuning_type gpf-plus --num_layers 3 --pnum 20

#----------------------------------------------------------------------------------------------------------------------------------------

## ClinTox

### Infomax

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/infomax.pth --dataset clintox --tuning_type gpf --num_layers 3

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/infomax.pth --dataset clintox --tuning_type gpf-plus --num_layers 3 --pnum 20

### EdgePred

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/edgepred.pth --dataset clintox --tuning_type gpf --num_layers 4

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/edgepred.pth --dataset clintox --tuning_type gpf-plus --num_layers 3 --pnum 10


### AttrMasking

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/masking.pth --dataset clintox --tuning_type gpf --num_layers 4

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/masking.pth --dataset clintox --tuning_type gpf-plus --num_layers 3 --pnum 10

### ContextPred

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/contextpred.pth --dataset clintox --tuning_type gpf --num_layers 3

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/contextpred.pth --dataset clintox --tuning_type gpf-plus --num_layers 3 --pnum 10

### GCL

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/gcl.pth --dataset clintox --tuning_type gpf --num_layers 1

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/gcl.pth --dataset clintox --tuning_type gpf-plus --num_layers 3 --pnum 5

#----------------------------------------------------------------------------------------------------------------------------------------

## MUV

### Infomax

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/infomax.pth --dataset muv --tuning_type gpf --num_layers 2

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/infomax.pth --dataset muv --tuning_type gpf-plus --num_layers 2 --pnum 20

### EdgePred

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/edgepred.pth --dataset muv --tuning_type gpf --num_layers 2

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/edgepred.pth --dataset muv --tuning_type gpf-plus --num_layers 2 --pnum 5

### AttrMasking

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/masking.pth --dataset muv --tuning_type gpf --num_layers 2

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/masking.pth --dataset muv --tuning_type gpf-plus --num_layers 1 --pnum 5

### ContextPred

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/contextpred.pth --dataset muv --tuning_type gpf --num_layers 2

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/contextpred.pth --dataset muv --tuning_type gpf-plus --num_layers 2 --pnum 5

### GCL

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/gcl.pth --dataset muv --tuning_type gpf --num_layers 4

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/gcl.pth --dataset muv --tuning_type gpf-plus --num_layers 1 --pnum 5

#----------------------------------------------------------------------------------------------------------------------------------------

## HIV

### Infomax

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/infomax.pth --dataset hiv --tuning_type gpf --num_layers 4

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/infomax.pth --dataset hiv --tuning_type gpf-plus --num_layers 3 --pnum 20

### EdgePred

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/edgepred.pth --dataset hiv --tuning_type gpf --num_layers 4

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/edgepred.pth --dataset hiv --tuning_type gpf-plus --num_layers 3 --pnum 5

### AttrMasking

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/masking.pth --dataset hiv --tuning_type gpf --num_layers 3

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/masking.pth --dataset hiv --tuning_type gpf-plus --num_layers 3 --pnum 5

### ContextPred

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/contextpred.pth --dataset hiv --tuning_type gpf --num_layers 3

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/contextpred.pth --dataset hiv --tuning_type gpf-plus --num_layers 3 --pnum 10

### GCL

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/gcl.pth --dataset hiv --tuning_type gpf --num_layers 2

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/gcl.pth --dataset hiv --tuning_type gpf-plus --num_layers 4 --pnum 5

#----------------------------------------------------------------------------------------------------------------------------------------

## BACE

### Infomax

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/infomax.pth --dataset bace --tuning_type gpf --num_layers 1

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/infomax.pth --dataset bace --tuning_type gpf-plus --num_layers 2 --pnum 5

### EdgePred

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/edgepred.pth --dataset bace --tuning_type gpf --num_layers 4

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/edgepred.pth --dataset bace --tuning_type gpf-plus --num_layers 3 --pnum 10

### AttrMasking

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/masking.pth --dataset bace --tuning_type gpf --num_layers 2

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/masking.pth --dataset bace --tuning_type gpf-plus --num_layers 3 --pnum 20


### ContextPred

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/contextpred.pth --dataset bace --tuning_type gpf --num_layers 4

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/contextpred.pth --dataset bace --tuning_type gpf-plus --num_layers 3 --pnum 5

### GCL

#### GPF

python prompt_tuning_few_shot.py --input_model_file pretrained_models/gcl.pth --dataset bace --tuning_type gpf --num_layers 1

#### GPF-plus

python prompt_tuning_few_shot.py --input_model_file pretrained_models/gcl.pth --dataset bace --tuning_type gpf-plus --num_layers 4 --pnum 10

#----------------------------------------------------------------------------------------------------------------------------------------











