# EBM-HEP

## Energy-based probabilistic modeling of high-energy events

Code repository for ["Versatile Energy-Based Probabilistic Models for High Energy Physics"](https://arxiv.org/abs/2302.00695), Taoli Cheng and Aaron Courville. 

### Abstract

As a classical generative modeling approach, energy-based models have the natural advantage of flexibility in the form of the energy function. Recently, energy-based models have achieved great success in modeling high-dimensional data in computer vision and natural language processing. 
In line with these advancements, we build a multi-purpose energy-based probabilistic model for High Energy Physics events at the Large Hadron Collider.  This framework builds on a powerful generative model and describes higher-order inter-particle interactions.
It suits different encoding architectures and builds on implicit generation. As for applicational aspects, it can serve as a powerful parameterized event generator for physics simulation, a generic anomalous signal detector free from spurious correlations, and an augmented event classifier for particle identification. 

![](EBM-Schematic.jpg)

### Datasets
Training sets and test sets for anomaly detection can be found [here](https://zenodo.org/records/4641460) and [here](https://zenodo.org/records/4614656).

### Training

#### Standard EBM

Input arguments

```
--input_dim: input dimension of each jet, i.e., 4*number of jet constituents
--input_scaler: scale the input features or not

--steps: MCMC steps
--step_size: MCMC step size
--epsilon: noise magnitude in Langevin Dynamics
--kl: add the KL divergence between the model distribution and MCMC estimation to the training objective
--hmc: use Hamiltonian Monte Carlo or not

--n_train: number of training samples
--batch_size: batch size
--epochs: number of training epochs
--lr: learning rate

--model_name: model name
```

Training

```
./ebm_jet_attn.py --input_dim 160 --steps 24 --step_size 0.1 --n_train 300000 --batch_size 128 --epochs 50 --lr 0.0001
```

### Citation
```
@inproceedings{
cheng2023versatile,
title={Versatile Energy-Based Probabilistic Models for High Energy Physics},
author={Taoli Cheng and Aaron Courville},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=j0U6XJubbP}
}
```
