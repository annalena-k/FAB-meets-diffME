# Flow Annealed importance sampling Bootstrap (FAB) meets differentiable Matrix Elements (ME)
This repository contains the code for the research paper:

> A. Kofler, V. Stimper, M. Mikhailenko, M. Kagan, L. Heinrich
> "Flow Annealed Importance Samling Bootstrap meets Differentiable Particle Physics." 
> Accepted at the _NeurIPS 2024 Machine Learning and the Physical Sciences Workshop_ and selected for a 🏆*spotlight contributed talk*!
> Available on [arXiv]().

---

## ⚛️ What is the goal of this paper?


## 🚀 Quickstart

To install the code in this repository, clone the code with
```bash
git clone TODO
```
and create a new virtual environment:
```bash
python -m venv venv-fab-meets-diffme
```
If you want to make sure that the `venv` picks up the correct `cuda` and `cudnn` versions installed on your machine or a cluster machine, use
```bash
python -m venv --system-site-packages venv-fab-meets-diffme
```
After activating the environment via
```bash
source venv-fab-meets-diffme/bin/activate
```
you need to enter the cloned directory
```bash
cd FAB-meets-diffME
```
In case of a GPU-capable machine, I can recommend first installing `JAX` following the instructions outlined on the [official webpage](https://jax.readthedocs.io/en/latest/installation.html) since the exact command depends on your local `cuda` version and might change with a new release.
Tipp:`FAB` depends on a specificy `blackjax-nightly` release which is not compatible with every `JAX` version. If you run into dependency issues (like me), you might need to specify the `JAX` version explicitely. Releases that worked for me in the past are `jax==0.4.13 jaxlib==0.4.13` and `jax==0.4.23 jaxlib==0.4.23`.

Finally, you can install (the rest of) the code with
```bash
pip install -e .
```
or for the developmental install:
```bash
pip install -e ."[dev]"
```

## 📜 Citation

If you find this code useful, please cite our paper:

```bibtex
@article{Kofler_2024,
  author     = {Kofler, Annalena and Stimper, Vincent and Mikhail, Mikhasenko and Kagan, Michael and Heinrich, Lukas},
  title      = {Flow Annealed Importance Sampling Bootstrap meets Differentiable Particle Physics},
  year       = 2024,
  journal    = {},
  eprint     = {},
  eprinttype = {arXiv},
  addendum   = {},
}
```
If you run FAB, please cite the original paper:
