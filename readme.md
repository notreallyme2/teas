# Target-Embedding Autoencoders
Models drawn from xxx et al.  
  
## Results
All results are MSE on a held-out test set

| Dataset        | Model     | MSE      |
| :------------- | :-------  | :------  |
|  sklearn       | Lasso     | 0.07288  |
| L1000 / GTEX   | Lasso     | tba      | |

## Notes  
* GeneNetWeaver utilities are not working yet  

## To do
1. Move functions to .py files (use Fire)
2. Train all models on sklearn data

## Repository structure
|–teas  
––|––data  
––––|––aggregate_gnw_data
––––|––skl_synthetic  
––––|––datasets  
––|––models  
––––|––linear_models  
––––|––nonlinear_models  
––––|––lasso_baseline


## Data sources:  
https://www.synapse.org/#!Synapse:syn2787209/wiki/70351
http://gnw.sourceforge.net/genenetweaver.html

