


## Data sources:  
https://www.synapse.org/#!Synapse:syn2787209/wiki/70351
http://gnw.sourceforge.net/genenetweaver.html



*From Dan:*

"Hi Lindsay,

After looking at my code, I realized it is probably easier to do this from scratch (using Algorithm 1 in the draft paper). I hadn’t been allowed any time to document things…

At the moment my code is heavily oriented towards time series. Attached is a linear example; I’ve pared this done and moved some stuff around so it is (slightly) easier to look at. I believe only the following two things are relevant in the static setting:

Graph is in lines 83-93. HAT means \hat{}, and TIL means \tilde{}; these respectively refer to the prediction and reconstruction versions of each variable.

Optimizers are in lines 151-186, and 414-448. This should correspond to Algorithm 1. For each solver, the XXX binary notation denotes predictor, encoder, and decoder. For instance, “solver_101” trains the predictor and decoder, but not the encoder. Direct prediction (without target-autoencoding) is simply training using “solver_101” (instead of the 3 stages using 011, 100, then 111).

These two things are probably the only things really relevant here. Everything else is just getting the dimensions of the time series to and from various orders, and computing time-series specific averages for performance metrics, dividing variables into static, temporal, binary, continuous, dataset-specific variable indexing, etc.

Again, I think if we’re using any static data, there is little here of use, and it’s much easier to start over…

Cheers,
Dan"
