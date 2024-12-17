# Model Selection for $f\sigma_8$ growth rate data sing Interpretable Machine Learning
![License Badge](https://img.shields.io/badge/license-MIT-brightgreen.svg)

## Table of Contents

- [Overview](#overview)
- [Description](#Description)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

---

## Overview


This repository hosts the codebase developed for the machine learning analysis presented in [arXiv:2406.08351](https://arxiv.org/abs/2406.08351). The objective is to perform model selection between ΛCDM and the Hu-Sawicky (HS) f(R) modified gravity model using simulated growth rate $f\sigma_8$ data. The workflow integrates advanced ML techniques with a focus on Local Interpretability via [LIME](https://github.com/marcotcr/lime.git) (Local Interpretable Model-agnostic Explanations), providing insights into the decision-making process of the machine learning models. This folder contains:
1. The Neural Network for model selection HS and ΛCDM. (jupyter notebook)
2. The application of [LIME](https://github.com/marcotcr/lime.git) for tabular data. (jupyter notebook)

## Installation

### Prerequisites

- Required software: `python`
- Dependencies: `numpy`, `matplotlib`, `tensorflow`, `class`, `lime`

### How to get started

```bash
# Example to get it running
pip install numpy matplotlib tensorflow lime
git clone https://github.com/IndiraOcampo/Growth_LSS_model_selection_Lime.git
```

## Usage

If you are using the content provided in this repository to do your own analysis, please cite this repository and the manuscript:

```bash
@misc{[Growth_LSS_model_selection_Lime.git,
  authors = {Ocampo I},
  title = {Growth LSS model selection and interpretability},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/IndiraOcampo/Growth_LSS_model_selection_Lime.git}},
}
```
```bash
@misc{ocampo2024enhancing,
  title={Enhancing Cosmological Model Selection with Interpretable Machine Learning},
  author={Ocampo, Indira and Alestas, George and Nesseris, Savvas and Sapone, Domenico},
  journal={arXiv preprint arXiv:2406.08351},
  year={2024},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE]() file for details.
