 # CTL

 This code gives the implementation  of the paper "Multi-Label Learning with Multiple Complementary Labels". The paper and appendix can be found at [here](https://gaoyi439.github.io/papers/TPAMI2024_without_acknowledgement.pdf).

 Requirements
- Python >=3.6
- PyTorch >=1.10

---
## Run:
**main.py**
- This is main function. After running, you will see a .csv file with the results saved in the directory.
The results will have seven columns: epoch number, training loss, hamming loss of test data, one error of test data,
coverage of test data, ranking loss of test data and average precision of test data.

## Specify the loss function argument:
- *args.lo=mcll_ctl*: $\mathcal{\bar L}_\text{CTL}$
- *args.lo=mcll_mae*: $\mathcal{\bar L}_\text{MAE}$
- *args.lo=mcll_bce*: $\mathcal{\bar L}_\text{BCE}$
## Specify the dataset argument:
- scene: scene dataset
- ml_tmc2007: tmc2007 dataset
