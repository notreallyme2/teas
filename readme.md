# Target-Embedding Autoencoders

Models drawn from xxx et al.  

## Notes

1. Install dependencies: `pipenv install`
2. Check that the tests run and pass: `python -m pytest`
3. Check that the training scripts run ok (`python3 train_xxx.py`)

**NB Data must always be normalised.**  
I'm not doing it in the `train_xxx.py` scripts, because the data are synthetically generated and normalised by default.  

When running `train_lasso_baseline.py` the plots of $X$ *and* $Y$ *variance explained* are checking that inputs and outputs are compressible (or why bother with a TEAS model!?).  
Other commented-out plots are legacy from ealier work.  

GeneNetWeaver utilities are not working yet  

### On building up a TEA

* First, use an AE to encode $Y \to Z \to Y$  
* Then train an MLP: $f: X \to Z$  
* Look at the performance of $X \to Z \to Y$  
* Transfer these weights into a full implementation of a TEA

## Results

All results are MSE on a held-out test set. You should get similar results by running the `train_xxx.py` scripts for each model.  

| Dataset        | Model         | MSE      |
| :------------- | :-------      | :------  |
| sklearn        | Lasso         | 0.03996  |
| sklearn        | Linear MLP    | 0.01569  |
| sklearn        | Linear FEA    | 0.01427  |
|sklearn         | Linear TEA    | 0.01429  ||

## To do

1. Train all models on sklearn data
2. Refactor (doing this now):  
   1. Get rid of .ipynb files
   2. Put all variables into .json files
   3. Create logging

## Repository structure

|–teas/  
––|––train_mlp.py  
––|––train_linear_feas.py  
––|––linear_teas.ipynb  
––|––test_models.py  
––|––test_data_utilities.py  
––|––data/  
––––|––aggregate_gnw_datacc
––––|––skl_synthetic  
––––|––datasets  
––|––models/  
––––|––linear  

## Data sources  

https://www.synapse.org/#!Synapse:syn2787209/wiki/70351
http://gnw.sourceforge.net/genenetweaver.html
