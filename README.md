# nhl-predictor

A statistical model which can predict winners in NHL games. It is based on the (homogenous)Poisson distribution and utilizes data gathered from the 2019-2020 NHL season. 
It performs regression analysis to come up with relative team strengths, taking into account home-ice advantage, and feeds the results to a match simulator which
produces a probability matrix; from there the probabilities of the Home Team winning, Away Team winning, or an Overtime/Shootout 
result can be calculated using simple matrix operations.

Probability predictions for the 2020 NHL Playoffs generated by the model:

![](images/stats1.PNG)

![](images/stats2.PNG)

Linear Regression results showing Offensive/Defensive strength coefficients of each NHL Team:

![](images/regression1.PNG)

![](images/regression2.PNG)


Requirements (Py modules): pandas, matplotlib, numpy, statsmodels, seaborn

To Run (Windows): 

py sports.py

