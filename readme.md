# Target-Embedding Autoencoders

An implementation of Target Embedding Autoencoders using Pytorch. As described in ["Target-Embedding Autoencoders for Supervised Representation Learning" by Jarret and van der Schaar](https://openreview.net/forum?id=BygXFkSYDH)

## How to use

These scripts will run out of the box on synthetically generated data. To generate the data, run `python3 data/skl_synthetic.py`

## Install notes

1. Clone the repo
2. Make sure you have [pytest](https://pypi.org/project/pytest/) and [pipenv](https://pipenv.kennethreitz.org/en/latest/) installed
3. We use `pipenv` to manage dependencies. To install all the things, run: `pipenv install` from the repo's root directory
4. Create the synthetic dataset by running `python3 data/skl_synthetic.py`
5. Check that the tests run and pass: `python -m pytest`
6. Check that the training scripts run ok (`python3 train_xxx.py`)
7. The TEAs model example is in `linear_teas.ipynb`

### Using your own data

* **NB Data must always be normalized.**  
* The synthetic data comes pre-normalised, but if running these models on data in the wild, ensure it is normalized first  

## Results

All results are MSE on a held-out test set. You should get similar results by running the `train_xxx.py` scripts for each model.  

| Dataset        | Model         | MSE      |
| :------------- | :-------      | :------  |
| sklearn        | Lasso         | 0.03996  |
| sklearn        | Linear MLP    | 0.01569  |
| sklearn        | Linear FEA    | 0.01427  |
|sklearn         | Linear TEA    | 0.01429  |

## To do

1. Create a `nonlinear.py` version with ReLUs