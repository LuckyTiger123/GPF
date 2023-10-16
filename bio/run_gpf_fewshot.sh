#### GIN fine-tuning
split=scaffold
dataset=$1

for runseed in 0 1 2 3 4
do
python prompt_tuning_few_shot.py --input_model_file models_graphcl/graphcl_80.pth --split $split --runseed $runseed --gnn_type gin --dataset $dataset --lr 1e-4 --dropout_ratio 0 --epochs 50 --tuning_type gpf --num_layers 4
done