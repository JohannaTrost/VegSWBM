# VegSWBM

Evapotranspiration (ET) is a main driver of water fluxes in ecosystems and is essential for modelling soil moisture dynamics (4). ET can be partly described as a function of vegetation (3, 4). The seasonal patterns observed in ET, driven by factors such as transpiration capacity dependent on seasonally varying leaf area index (LAI) and stomatal conductance, present a crucial aspect of ecosystem water exchange dynamics (2, 4). The simple water balance model (SWBM) by Orth et al. (2013) does not account for these seasonal patterns.
Here, we use a sinusoidal curve to modulate the maximum ET (denoted β₀ in the SWBM by Orth et al. (2013)) that limits the conversion of net radiation into ET accounting for e.g. water transport through vegetation.

We show that introducing seasonal variation of vegetation we could marginally improve seasonal ET patterns in different sites. However, this improvement is not necessarily linked to an improvement in modeled runoff and soil moisture patterns. 

## Install
1. `cd <path/to/where/you/want/the/repo>`
2. `git clone https://github.com/JohannaTrost/VegSWBM.git`
3. `cd VegSWBM`
4. `pip install -r requirements`

## Notebooks

**results.ipynb**

Here we present the correlations of runoff, soil moisture and evapotranspiration of our final calibrated model
and ERA5 reanalysis data. We further compare VegSWBM to the original SWBM. In addition, we plot the 2018 time series of 
VegSWBM, SWBM and the ERA5 data.

**calibration.ipynb and non_seasonal_calibration.ipynb**

In these notebooks we calibrated VegSWBM and the non-seasonal SWBM. For this we performed grid search over the following
parameter space:

- c_s: 210, 420, 840 (water holding capacity)
- b0: 0.4, 0.6, 0.8 (limit of ET or max. ET)
- γ: 0.2, 0.5, 0.8 (shape parameter of the ET function)
- α: 2, 4, 8 (shape parameter of the runoff function)

Note that for VegSWBM we used the b0 values as the initial center value of the sine curve, which was then passed to 
the optimizer (scipy's minimize function). The optimizer then maximized the correlation of ET of the model and ERA5 data.
The remaining initial values of the sine curve were not calibrated (amplitude = 0.5, frequency = 2, phase = 5).
For calibration we ran the model from 2008 to 2013 and computed the correlation during 2009 to 2013 (using 2008 as spin up year).

**model_test.ipynb**

For the validation we ran the model with the best parameter set during 2013 to 2018 and computed the correlations during 
2014 to 2018. The resulting model time series from 2013 to 2018 was then stored for the original SWBM and for our VegSWBM (in data/output).
The resulting correlation values are available in the results folder. 

**param_selection.ipynb**

Primary to all the above, we made the choice of only making B0 (ET limit) seasonal. We computed the partial dependence 
of the seasonality for B0 and all other SWBM model parameters and found that making B0 seasonal did yield the most 
important improvement.



