# Efficient-KD

You should download the dataset CoNLL-03 first and save it into './datasets/conll-03' first.

Train the teacher model or baseline (bert_layers=12 as the teacher in the script): 
```
CUDA_VISIBLE_DEVICES=0 python train_bert_crf.py
```

Train the Efficient KD (You might set the corresponding teacher model path): 
```
CUDA_VISIBLE_DEVICES=0 python train_distil_bert_crf.py
```