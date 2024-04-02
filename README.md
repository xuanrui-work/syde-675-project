# uda-empirical-survey

This repository contains the pytorch implementation of some simple and/or popular approaches to deep unsupervised domain adaptation (UDA), and the empirical results of these approaches on image classification tasks.

The following papers are implemented and explored:
* [Correcting Sample Selection Bias by Unlabeled Data (CSSB)](https://papers.nips.cc/paper_files/paper/2006/hash/a2186aa7c086b46ad4e8bf81e2a3a19b-Abstract.html)
* [Deep Domain Confusion: Maximizing for Domain Invariance (DDC)](https://arxiv.org/abs/1412.3474)
* [Deep CORAL: Correlation Alignment for Deep Domain Adaptation (CORAL)](https://arxiv.org/abs/1607.01719)
* [Domain-Adversarial Training of Neural Networks (DANN)](https://arxiv.org/abs/1505.07818)
* [Adversarial Discriminative Domain Adaptation (ADDA)](https://arxiv.org/abs/1702.05464)

## Getting started

### Prerequisites
Run the below command to install all the required python dependencies.
```
pip install -r requirements.txt
```

### Usage
Simply open the corresponding jupyter notebook with your favorite editor and run the cells. Training logs will be displayed through tensorboard, which can be started by running the below command:
```
tensorboard --logdir=<dir-to-logs>
```

## Results
We tune each approach using the target validation set, afterwhich we perform 5 runs of each approach with random initializations and report the average test accuracy and standard error on the target test set. For each run, the best model is selected based on the target validation set and tested upon. The results are summarized in the table below:

|               | MNIST -> USPS | USPS -> MNIST |
|---------------|---------------|---------------|
| Src-Only      | 80.08 ± 0.80% | 73.25 ± 1.57% |
| Tgt-Only      | 95.91 ± 0.27% | 99.04 ± 0.07% |
| Src&Tgt       | 96.71 ± 0.12% | 99.00 ± 0.06% |
|---------------|---------------|---------------|
| CSSB          | 84.25 ± 0.28% | 64.23 ± 2.52% |
| DDC           | 86.65 ± 0.15% | 79.09 ± 0.72% |
| CORAL         | 89.24 ± 0.53% | 88.92 ± 0.56% |
| DANN          | 90.72 ± 0.45% | 90.01 ± 1.24% |
| ADDA          | 91.36 ± 0.35% | 88.05 ± 1.64% |

