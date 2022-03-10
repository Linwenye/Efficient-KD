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



## Cite

If you find that efficient sub-structured KD is helpful for your paper , please cite our [paper](https://arxiv.org/abs/2203.04825):

```

@misc{lin2022efficient,
    title={Efficient Sub-structured Knowledge Distillation},
    author={Wenye Lin and Yangming Li and Lemao Liu and Shuming Shi and Hai-tao Zheng},
    year={2022},
    eprint={2203.04825},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

