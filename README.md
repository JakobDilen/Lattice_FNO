# Fourier Neural Operator Learning Using Rank-1 Lattices

## Lattice FNO

The lattice FNO implementation can be found in on [Github](https://github.com/dnuy/neuraloperator.git@feature/index-sets-lattice-fft). In order to more easily train the models, some helper functions can be found in `src/utility`. In order to run these scripts, it is necessary to install the packages in `requirements.txt`. The reason for using `PyTorch==2.5.0` is due to compatibility issues with the `neuralop` package for higher Pytorch versions.

In order for the experiments to work, some changes to the `neuralop` package are necessary.
These changes are implemented in the fork [dnuy/neuralop](https://github.com/dnuy/neuraloperator.git@feature/index-sets-lattice-fft)

## Experiments

To run all experiments, simply run the `src/run_all_scripts.py` file. This will automatically run all the necessary experiments, assuming that the correct packages are installed.

## Data generation

The `src/fourier_data_generation` folder contains the necessary python files to generate the data. 