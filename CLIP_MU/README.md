# CLIP Unlearning Scenario

## Dowload Pretrained and Fine-tuned Weights
Fine-tuned model checkpoints and their corresponding classifier heads, are available at [this link](https://drive.google.com/drive/u/1/folders/1m1iHi5KoTN1Fg5JqIZxtVP1ZTxgILZyi).

## Dataset
We evaluate across eight benchmarks: SUN397, Stanford Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, and DTD.
For details on dataset preprocessing, please refer to the [Task Arithmetic](https://github.com/mlfoundations/task_vectors) or [Tangent Arithmetic](https://github.com/gortizji/tangent_task_arithmetic/tree/main).

## Dependencies 
1. Install necessary packages:
```sh
conda env create
conda activate negmerge-clip
```
2. Add the src directory to your Python path:
```sh
cd CLIP_MU
export PYTHONPATH="$PYTHONPATH:$PWD"
```

## Training
### Finetune the CLIP Model
```python
# Finetune non-linearly on 2 GPUs
python src/finetune.py --finetuning-mode=standard --model=ViT-B-32 --world-size=2

# Finetune linearly on 2 GPUs
python src/finetune.py --finetuning-mode=linear --model=ViT-B-32 --world-size=2

# Finetune non-linearly with randaug
python src/finetune.py --finetuning-mode=standard --model=ViT-B-32 --world-size=2 --auto-aug "rand-m1-n1-mstd0.5"
```

## Evaluation
### Single-task Accuracy
```python
# Evaluate pre-trained models.
python src/eval_single_task.py --model=ViT-B-32 --finetuning-mode=none

# Evaluate non-linearly fine-tuned models.
python src/eval_single_task.py --model=ViT-B-32 --finetuning-mode=standard

# Evaluate linearly fine-tuned models.
python src/eval_single_task.py --model=ViT-B-32 --finetuning-mode=linear
```
### Task Negation
```python
# Evaluate non-linearly fine-tuned models.
python src/eval_task_negation.py --model=ViT-B-32 --finetuning-mode=standard

# Evaluate linearly fine-tuned models.
python src/eval_task_negation.py --model=ViT-B-32 --finetuning-mode=linear
```

### NegMerge
```python
# Evaluate non-linear NegMerge model.
python src/negmerge.py --model=ViT-B-32 --finetuning-mode=standard

# Evaluate linear NegMerge model.
python src/negmerge.py --model=ViT-B-32 --finetuning-mode=linear
```

### Reference
https://github.com/gortizji/tangent_task_arithmetic/tree/main
