# EF-VFL

This repository contains the code used for the experiments in the paper:  
**[EF-VFL: Communication-efficient Vertical Federated Learning via Compressed Error Feedback](https://arxiv.org/abs/2406.14420)**.

Pedro Valdeira, João Xavier, Cláudia Soares, Yuejie Chi.

### Abstract

Communication overhead is a known bottleneck in federated learning (FL). To address this, lossy compression is commonly used on the information communicated between the server and clients during training. In horizontal FL, where each client holds a subset of the samples, such communication-compressed training methods have recently seen significant progress. However, in their vertical FL counterparts, where each client holds a subset of the features, our understanding remains limited. To address this, we propose an error feedback compressed vertical federated learning (EF-VFL) method to train split neural networks. In contrast with previous communication-compressed methods for vertical FL, EF-VFL does not require a vanishing compression error for the gradient norm to converge to zero for smooth nonconvex problems. By leveraging error feedback, our method can achieve a $\mathcal{O}(1/T)$ convergence rate in the full-batch case, improving over the state-of-the-art $\mathcal{O}(1/\sqrt{T})$ rate under $\mathcal{O}(1/\sqrt{T})$ compression error, and matching the rate of uncompressed methods. Further, when the objective function satisfies the Polyak-Łojasiewicz inequality, our method converges linearly. In addition to improving convergence rates, our method also supports the use of private labels. Numerical experiments show that EF-VFL significantly improves over the prior art, confirming our theoretical results.

### Usage

To set up the environment, run the following command:

```bash
conda env create -f environment.yaml
```

Next, activate the environment:

```bash
conda activate EFVFL
```

Lastly, to run an experiment, simply run main.py with the appropriate arguments. For example:

```bash
python main.py --config mnist_fullbatch/svfl.yaml --gpu 0 --seeds 0 1 2 3 4
```

### Citation

If you find this repository useful, please cite [our work](https://arxiv.org/abs/2406.14420):

```bibtex
@article{valdeira2024communication,
  title={Communication-efficient Vertical Federated Learning via Compressed Error Feedback},
  author={Valdeira, Pedro and Xavier, Jo{\~a}o and Soares, Cl{\'a}udia and Chi, Yuejie},
  journal={arXiv preprint arXiv:2406.14420},
  year={2024}
}
```