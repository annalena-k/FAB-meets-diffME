# Flow Annealed importance sampling Bootstrap (FAB) meets Differentiable Matrix Elements (ME)
![Python 3.10](https://img.shields.io/badge/python-3.10+-blue)
[![Data availability](https://img.shields.io/badge/Data-Available_on_Edmond-31705e)](https://doi.org/10.17617/3.UZ786R)

This repository contains the code for the research paper:

> A. Kofler, V. Stimper, M. Mikhailenko, M. Kagan, L. Heinrich
> "Flow Annealed Importance Samling Bootstrap meets Differentiable Particle Physics." 
> Accepted at the _Machine Learning and the Physical Sciences Workshop, NeurIPS 2024_ 
> 🏆 Selected for a *spotlight contributed talk* (best 2%)!

---

## 🚀 Quickstart

To install the code in this repository, clone the code with
```bash
git clone git@github.com:annalena-k/FAB-meets-diffME.git
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
and (potentially) upgrading `pip` with `pip install --upagrade pip`, you need to enter the cloned directory
```bash
cd FAB-meets-diffME
```
In case of a GPU-capable machine, I recommend first installing `JAX` following the instructions outlined on the [official webpage](https://jax.readthedocs.io/en/latest/installation.html) since the exact command depends on your local `cuda` version and might change with a new release.

Finally, you can install (the rest of) the code with
```bash
pip install -e .
```
or for the developmental install:
```bash
pip install -e ."[dev]"
```

## 📈 Data
The data used for training the model with the forward KL divergence are available via [Edmond](https://doi.org/10.17617/3.UZ786R).
The samples for $\Lambda_c^+ \rightarrow pK^-\pi^+$ were generated with rejection sampling, while the samples for $e^+e^- \rightarrow t\bar{t}, t\rightarrow W^+ b, \bar{t} \rightarrow W^- \bar{b}$ were obtained via `MadGraph`.

## 📜 Citation

If you find this code useful, please cite our paper:

> Annalena Kofler, Vincent Stimper, Mikhail Mikhasenko, Michael Kagan, Lukas Heinrich.
> Flow Annealed Importance Sampling Bootstrap meets Differentiable Particle Physics. Machine Learning and the Physical Sciences Workshop, NeurIPS 2024.


```bibtex
@article{Kofler_2024,
  author     = {Kofler, Annalena and Stimper, Vincent and Mikhasenko, Mikhail and Kagan, Michael and Heinrich, Lukas},
  title      = {Flow Annealed Importance Sampling Bootstrap meets Differentiable Particle Physics},
  year       = 2024,
  journal    = {},
  eprint     = {},
  eprinttype = {arXiv},
  addendum   = {},
}
```
If you run FAB, please cite the original paper:

> Laurence I. Midgley, Vincent Stimper, Gregor N. C. Simm, Bernhard Schölkopf, José Miguel Hernández-Lobato.
> Flow Annealed Importance Sampling Bootstrap. The Eleventh International Conference on Learning Representations. 2023.

```
@inproceedings{
midgley2023flow,
title={Flow Annealed Importance Sampling Bootstrap},
author={Laurence Illing Midgley and Vincent Stimper and Gregor N. C. Simm and Bernhard Sch{\"o}lkopf and Jos{\'e} Miguel Hern{\'a}ndez-Lobato},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=XCTVFJwS9LJ}
}
```

## 📎 Remarks about the versioning
`FAB` depends on a specific `blackjax-nightly` release which is not compatible with every `JAX` version. If you run into dependency issues (like I did), you need to specify the `JAX` version explicitely which is already included in the `pyproject.toml` file. Releases that worked for me in the past are `jax==0.4.13 jaxlib==0.4.13` and `jax==0.4.23 jaxlib==0.4.23`.
The reason is that `jax.random.PRNGKeyArray` (used by this `blackjax` version) is [removed in JAX v0.4.24](https://stackoverflow.com/questions/78302031/stable-diffusion-attributeerror-module-jax-random-has-no-attribute-keyarray).
Additionally, these JAX versions are only compatible with numpy v1.x due to [changes in the `copy` keyword for v2.0](https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword) and `scipy` v.11 because `scipy.linalg.tril` gets removed in v.12+. 
Since numpy v1.x uses `pkgutil` during installation with `pip` which is removed in python v.3.12., this package can only run with python 3.11. or earlier.
At the time of publication (November 2024), installation with `pip install -e .` should be working without additional changes.

For reference and reproducibility, I included a screenshot of the packages installed in the environment with which the results were created named `requirements_screenshot_20241117.txt`.
