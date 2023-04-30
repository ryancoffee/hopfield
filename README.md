# Hopfield Network Debugging Repo

In this repo, we will be exporing Hopfield networkds as an alternative to the traditional approach to attention in transformer models.

## thoughts
Make two calsses of "molecules" with a couple gaussian spectral features.  
Use those features to select 5 random electron counts of the 256 nonuniform quantized bins.    
Try to use the Wasserstein distance to determine "similar" and "dis-similar" binary vectors.  
Split the similar vectors across the Hopfield heads and train each head from only a few of each dis-similar training sets.  
Then feed combinations of multiple molecule classes with ~25-50 counts in order to see if the multi-head hopfield can get the relative ratios right.
Give it a new molecule and see if it can recover the input, run it like an auto-encoder.

Later, try doing nonuniform quantization.

