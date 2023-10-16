#### GIN fine-tuning
split=scaffold
dataset=$1

for runseed in 0 1 2 3 4
do

if [$dataset == "bbbp"]; then
	python prompt_tuning_few_shot.py --input_model_file models_graphcl/graphcl_80.pth --split $split --runseed $runseed --gnn_type gin --dataset $dataset --lr 1e-3 --epochs 100 --tuning_type gpf --num_layers 3

elif [$dataset == "tox21"]; then
	python prompt_tuning_few_shot.py --input_model_file models_graphcl/graphcl_80.pth --split $split --runseed $runseed --gnn_type gin --dataset $dataset --lr 1e-3 --epochs 100 --tuning_type gpf --num_layers 1

elif [$dataset == "toxcast"]; then
	python prompt_tuning_few_shot.py --input_model_file models_graphcl/graphcl_80.pth --split $split --runseed $runseed --gnn_type gin --dataset $dataset --lr 1e-3 --epochs 100 --tuning_type gpf --num_layers 4

elif [$dataset == "sider"]; then
	python prompt_tuning_few_shot.py --input_model_file models_graphcl/graphcl_80.pth --split $split --runseed $runseed --gnn_type gin --dataset $dataset --lr 1e-3 --epochs 100 --tuning_type gpf --num_layers 4

elif [$dataset == "clintox"]; then
	python prompt_tuning_few_shot.py --input_model_file models_graphcl/graphcl_80.pth --split $split --runseed $runseed --gnn_type gin --dataset $dataset --lr 1e-3 --epochs 100 --tuning_type gpf --num_layers 1

elif [$dataset == "muv"]; then
	python prompt_tuning_few_shot.py --input_model_file models_graphcl/graphcl_80.pth --split $split --runseed $runseed --gnn_type gin --dataset $dataset --lr 1e-3 --epochs 100 --tuning_type gpf --num_layers 4

elif [$dataset == "hiv"]; then
	python prompt_tuning_few_shot.py --input_model_file models_graphcl/graphcl_80.pth --split $split --runseed $runseed --gnn_type gin --dataset $dataset --lr 1e-3 --epochs 100 --tuning_type gpf --num_layers 2

elif [$dataset == "bace"]; then
	python prompt_tuning_few_shot.py --input_model_file models_graphcl/graphcl_80.pth --split $split --runseed $runseed --gnn_type gin --dataset $dataset --lr 1e-3 --epochs 100 --tuning_type gpf --num_layers 1

fi

done