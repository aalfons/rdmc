# RMCLab: Lab for Matrix Completion and Imputation of Discrete Rating Data

Collection of methods for rating matrix completion, which is a statistical framework for recommender systems. Another relevant application is the imputation of rating-scale survey data in the social and behavioral sciences. Note that matrix completion and imputation are synonymous terms used in different streams of the literature. 

The main functionality implements robust matrix completion for discrete rating-scale data with a low-rank constraint on a latent continuous matrix. More information can be found in our paper:

Archimbaud, A., Alfons, A., and Wilms, I. (2025). Robust Matrix Completion for Discrete Rating-Scale Data. arXiv:2412.20802. doi:[10.48550/arXiv.2412.20802](https://doi.org/10.48550/arXiv.2412.20802).

In addition, the package provides wrapper functions for `softImpute` ([Mazumder, Hastie, and Tibshirani, 2010](https://www.jmlr.org/papers/v11/mazumder10a.html); [Hastie, Mazumder, Lee, Zadeh, 2015](https://www.jmlr.org/papers/v16/hastie15a.html)) for easy tuning of the regularization parameter, as well as benchmark methods such as median imputation and mode imputation.


## Installation

To install the latest version from GitHub, you can pull this repository and 
install it from the `R` command line via

```
install.packages("devtools")
devtools::install_github("aalfons/RMCLab")
```

If you already have package `devtools` installed, you can skip the first
line. Moreover, package `RMCLab` contains `C++` code that needs to be
compiled, so you may need to download and install the [necessary tools
for MacOS](https://cran.r-project.org/bin/macosx/tools/) or the
[necessary tools for
Windows](https://cran.r-project.org/bin/windows/Rtools/).


## Report issues and request features

If you experience any bugs or issues or if you have any suggestions for additional features, please submit an issue via the [*Issues*](https://github.com/aalfons/RMCLab/issues) tab of this repository.  Please have a look at existing issues first to see if your problem or feature request has already been discussed.


## Contribute to the package

If you want to contribute to the package, you can fork this repository and create a pull request after implementing the desired functionality.


## Ask for help

If you need help using the package, or if you are interested in collaborations related to this project, please get in touch with the [package maintainer](https://personal.eur.nl/alfons/).
