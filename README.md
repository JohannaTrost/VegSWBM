# VegSWBM

## Get the repo
`cd <path/to/where/you/want/the/repo>`

`git clone https://github.com/JohannaTrost/VegSWBM.git`

`cd VegSWBM`
### gespräch mit rene:
beta most impact?
first of all single seasonal parameter variation
c_s no seasonal variation (small?)

when beta is most important check other sites

## Final plots
Feature importance

Observed vs Initial vs Seasonal
    -> soil moisture
    -> runoff
    -> ET

optimized sinus function for Beta (in methods)
    

## maby:

first fit and optimize paramters for functions within Swbm 

## Approach:

-> seasonal variation for beta gamma alpha and c_s

-> seasonal variation in form of sinus function

-> optimizing sinus function parameters (amplitude, frequency, phase and offset) for best possible correlation between Swbm output and true values (soil moisture and evapotranspiration)

## to do:

-> compute **feature importance** e.g. with partical dependence (idea: predict with all seasonal and fixing one and report difference of correlation)

-> split data in test and training set (Rene fragen?)

-> if necessary adjust optimization for more "realism" ie. add conditions to function parameters (maximum amplitude)

-> fix cetrain sinus function paramters (ie. phase)

-> weighted importance of correlation for sm and et

-> test model on diffrent datasets 

-> test diffrent random seeds

-> check if range of seasonal variation of parameters are physicly possible **DONE**

## thoughts for the poster:
-> case study

-> introduction

-> why vertain parameters seasonal
-> try to explain what the seasonal variation does 
-> influence on model output

-> plot comparing model output and true values 
-> for all 3 datasets?

-> table with correlations

-> plot seasonal variation of certain parameters

-> compare sinus functions of parameters

-> plot influance of seasonal variation

-> outlook box

-> references with qr-code

-> check evaluation sheet
-> canva? softweare
-> space!
-> print poster in A4 at first 
-> print at 12.02 Kostenstelle 11000200211 format A0
-> check evaluation sheet 


## Tracking changes

1. `git status` - prints your changes and all necessary information.
2. `git add .` - will add all changes ("." means the current directory).
3. `git add <file>` or `git add <folder>` - adds only a specific file or folder.
4. `git status` - will now tell you that you have uncomitted changes or so.
5. `git commit <file> -m "Put a useful message about your changes"` - "commiting" is an essential step to track all changes and their history (<file> can also be a folder and if you don't specify it all the things you "added" will be commited). Things that have not been added (with `git add`) cannot be commited!
6. `git status` - if all you have committed all changes then `git status` will tell you.

Now, your changes are well tracked! But only on you local computer! You need to push them to the remote repo to make them available for everyone:

7. `git push`


