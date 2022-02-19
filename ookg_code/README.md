# SLAN
This code is modified based on https://github.com/thunlp/OpenKE

During the test time, we divide the test2id.txt into two parts: testsub.txt (ookg entities occur as subjects in the triples), testobj (ookg entities occur as objects in the triples).


## Reproducing results

In order to reproduce the results on the datasets in the paper, run the following commands

```
python train_slan.py --max_epochs 2000 --loss_weight 0.2 --dim 100 --margin 6.0 --rate 5.0
```



