# ToeplitzLDA

Code for the ToeplitzLDA classifier proposed in [here](https://arxiv.org/abs/2202.02001).
The classifier conforms sklearn and can be used as a drop-in replacement for other LDA
classifiers. For usage refer to the learning from label proportions (LLP) example or the
example script.

Note we used Ubuntu 20.04 with python3.8 to generate our results.

## Getting Started / User Setup

If you only want to use this library, you can use the following setup. Note that this
setup is based on a fresh Ubuntu 20.04 installation.

### Getting fresh ubuntu ready

```bash
apt install python3-pip python3-venv
```

### Python package installation

0. (Optional) Install fortran Compiler. On ubuntu: `apt install gfortran`
1. Create virtual environment: `python3 -m venv toeplitzlda_venv`
2. Activate virtual environment: `source toeplitzlda_venv/bin/activate`
3. Update pip: `pip install --upgrade pip`
4. Install numpy: `pip install numpy`
5. Install toeplitzlda: `pip install toeplitzlda[solver]`, if you dont have a fortran
   compiler: `pip install toeplitzlda`

### Check if everything works

Either clone this repo or just download the `scripts/example_toeplitz_lda.py` file and run
it: `python example_toeplitz_lda.py`. Note that this will automatically download EEG data
with a size of around 650MB.

## Development Setup

We use a fortran compiler to provide speedups for solving block-Toeplitz linear equation
systems. If you are on ubuntu you can install `gfortran`.

We use `poetry` for dependency management. If you have it installed you can simply use
`poetry install` to set up the virtual environment with all dependencies.

If setup does not work for you, please open an issue. We cannot guarantee support for many
different platforms, but could provide a singularity image.

## Learning from label proportions

Use the run_llp.py script to apply ToeplitzLDA in the LLP scenario and create results file
that can then be visualized using visualize_llp.py to create the plots shown in our
publication. Note that the two datasets will be downloaded automatically and are
approximately 16GB in size.

## ERP benchmark

This is not yet available.

Note this benchmark will take quite a long time if you do not have access to a computing
cluster. The public datasets (including the LLP datasets) total a size of approximately
120GB.

BLOCKING TODO: How should we handle the private datasets?
