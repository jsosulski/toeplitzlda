# ToeplitzLDA

This is an anonymized version of this repository without `.git`

Code for the ToeplitzLDA classifier proposed in (TODO insert reference).
The classifier conforms sklearn and can be used as a drop-in replacement for other LDA classifiers.
For usage refer to the learning from label proportions (LLP) example or the benchmark scripts.

Note we used Ubuntu 20.04 with python3.8 to generate our results.

## Setup

We use a fortran compiler to provide speedups for solving linear equation systems. If you are on ubuntu you can install `fortran-compiler` which finds the correct compiler for your system.

We use `poetry` for dependency management. If you have it installed you can simply use `poetry install` to set up the virtual environment with all dependencies.

If setup does not work for you, please open an issue. We cannot provide support for different platforms, but could provide a singularity image.

## Learning from label proportions

Use the run_llp.py script to apply ToeplitzLDA in the LLP scenario and create results file that can then be visualized using visualize_llp.py to create the plots shown in the publication at TODO.
Note that the two datasets will be downloaded automatically and are approximately 16GB in size.

## ERP benchmark

This is not yet available.

Note this benchmark will take quite a long time if you do not have access to a computing cluster.
The public datasets (including the LLP datasets) total a size of approximately 35GB.

BLOCKING TODO: How should we handle the private datasets?
