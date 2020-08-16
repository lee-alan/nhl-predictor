# nhl-predictor

A statistical model which can predict winners in NHL games. It is based on the (homogenous)Poisson distribution and utilizes data gathered from the 2019-2020 NHL season. 
It performs regression analysis to come up with relative team strengths, taking into account home-ice advantage, and feeds the results to a match simulator which
produces a probability matrix; from there the probabilities of the Home Team winning, Away Team winning, or an Overtime/Shootout 
result can be calculated using simple matrix operations.

Probability predictions for the 2020 NHL Playoffs generated by the model:

![](images/stats1.PNG)

![](images/stats2.PNG)

Linear Regression results showing Offensive/Defensive strength coefficients of each NHL Team:
(Take e^Coefficient to un-log the scale and obtain a coefficient which displays strength relative to average team, i.e e^0 = 1 = perfectly average)

![](images/regression1.PNG)

![](images/regression2.PNG)

Conclusion: 

This model is based off simple Poisson. This inherently is a drawback because goals scored in a games are not always independent. For example, in a 1 goal game with 1 minute left, the chance of scoring is likely higher. Another consideration is the accuracy of statistics. I used datasets from https://www.hockey-reference.com/leagues/NHL_2020_games.html. But goals scored can be deceiving in that sometimes teams can have injuries, very good games where they score an abnormal number of goals etc; this influences and skews results. Also, this model is a homogenous Poisson process, hence it assumes goals are scored uniformly throughout 3 periods, this is unlikely to be the case. Thus suggestions for improvement include utilizing non-homogenous Poisson processes, research suggests the Weibull distribution may be a good candidate.

Requirements (Py modules): pandas, matplotlib, numpy, statsmodels, seaborn

To Run (Windows): 

py sports.py

