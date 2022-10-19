# IDC
This repository contains the official implementation of the IDC Framework proposed in our paper "Input Distribution Coverage: Measuring Feature Interaction
Adequacy in Neural Network Testing".

## Datasets
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

## Components
1. Out-of-distribution data detection 
2. Variational Autoencoder
3. Combinatorial Coverage Measurement

### Out-of-distribution (OOD) data detection 
We used the Likelihood-Regret<sup>[[1]](#1)</sup> based out-of-distribution (OOD) filter in the framework. [Likelihood Regret: An Out-of-Distribution Detection Score For Variational Auto-encoder](https://github.com/XavierXiao/Likelihood-Regret) has the code and pretrained models available.

The included code is for measuring the total t-way coverage of the test datasets of MNIST, Fashion-MNIST and CIFAR10. Since the test datasets are assumed to be in-distribution data, we have not included the Likelihood Regret code base in this repository. When using this framework for other test sets, please follow the instructions from the Likelihood-Regret repository for filtering out the OOD inputs from the test sets.

### Variational Autoencoder (VAE)
FactoVAE<sup>[[2]](#2)</sup> and $\beta$-TCVAE<sup>[[3]](#3)</sup> models from [https://github.com/YannDubs/disentangling-vae](https://github.com/YannDubs/disentangling-vae) repository is used in this work.

### Combinatorial Coverage Measurement
Combinatorial Coverage Measurement (CCM) Command Line Tool from [https://github.com/usnistgov/combinatorial-testing-tools](https://github.com/usnistgov/combinatorial-testing-tools) is used for measuring the total t-way coverage of a test set.

Copy `ccmcl.jar` from `CCM Command Line Tool` directory of the repository to the current project directory. 
Running this tool requires Java installed on the machine.

## Usage
- Create virtual environment

    `python -m venv idc`
- Activate virtual environment

    `source idc/bin/activate`
- Install the required packages using the `requirements.txt` file.

    `pip install -r requirements.txt`

- Run `measure_coverage.py` to measure the total t-way coverage of the test datasets of MNIST, Fashion-MNIST and CIFAR10. Refer to [results](https://github.com/swa112003/IDC_temp/tree/main/results) directory for supported VAE models.

```
  python measure_coverage.py [vae] --dataset [mnist/fmnist/cifar10] --no_bins [intervals] --ways [ways] --density [target density: range[0,1]]

  E.g. : python measure_coverage.py btcvae_mnist_6 --dataset mnist --no_bins 20 --ways 3 --density 0.9999
```

## Measuring test coverage of custom test sets
Test coverage of custom test sets can be measured by setting the file path argument of `measure_coverage.py`. Use Likelihood Regret to filter out the OOD test inputs, and convert the in-distribution test inputs into numpy format. Run the below command.

```
python measure_coverage.py [vae] --dataset [mnist/fmnist/cifar10] --no_bins [intervals] --ways [ways] --density [target density: range[0,1]] --path [numpy file path]

E.g. python measure_coverage.py btcvae_mnist_6 --dataset mnist --no_bins 20 --ways 3 --density 0.9999 --path ./custom_testset.npy
```
## References
<a id="1">[1]</a> Xiao, Zhisheng, Qing Yan, and Yali Amit. "Likelihood regret: An out-of-distribution detection score for variational auto-encoder." Advances in neural information processing systems 33 (2020): 20685-20696.

<a id="2">[2]</a> Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising." International Conference on Machine Learning. PMLR, 2018.

<a id="3">[3]</a> Chen, Ricky TQ, et al. "Isolating sources of disentanglement in variational autoencoders." Advances in neural information processing systems 31 (2018).
