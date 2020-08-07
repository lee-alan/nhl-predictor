import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn

from scipy.stats import poisson, skellam

raw = pd.read_excel("Data.xlsx")
raw = raw[["Visitor","G","Home","G2"]].rename(columns = {"G" : "Visitor Goals", "G2" : "Home Goals"})

# Poisson model the goals - Poisson(H)\Poisson(R)
# Probability of Home team winning by X goals
# min gpg 3.16
# van gpg 3.25
#unused atm
print(raw.mean())
avg_home_goals = raw.mean()[0]
avg_visitor_goals = raw.mean()[1]
#print("Home GPG: ", avg_home_goals)
#print("Visitor GPG", avg_visitor_goals)
#print(skellam.pmf(0, avg_home_goals, avg_visitor_goals))

model_data = pd.concat( [raw[["Visitor", "Home", "Home Goals"]].assign(home="yes")
				.rename(columns = {"Home" : "Team", "Visitor" : "Opposition", "Home Goals" : "Goals"}),
					raw[["Visitor", "Home", "Visitor Goals"]].assign(home="no").rename(columns = 
						{"Visitor" : "Team", "Home" : "Opposition", "Visitor Goals" : "Goals"})])
# GLM , poisson
# Dep. var / scalar response : Goals
# Indep. var / expanatory : 
model = smf.glm(formula="Goals ~ home + Team + Opposition", data=model_data, family=sm.families.Poisson()).fit()
print(model.summary())
#unused atm
vancouver_score_home = model.predict(pd.DataFrame(data={"Team" : "Vancouver Canucks", "Opposition" : "Minnesota Wild", "home" : 'yes'},index=[1]))
minnesota_score_home = model.predict(pd.DataFrame(data={"Team" : "Minnesota Wild", "Opposition" : "Vancouver Canucks", "home" : 'yes'},index=[1]))
vancouver_score_away = model.predict(pd.DataFrame(data={"Team" : "Vancouver Canucks", "Opposition" : "Minnesota Wild", "home" : 'no'},index=[1]))
minnesota_score_away = model.predict(pd.DataFrame(data={"Team" : "Minnesota Wild", "Opposition" : "Vancouver Canucks", "home" : 'no'},index=[1]))
# van gf 228 , min gf 220
'''
print("Van goals @ home",vancouver_score_home)
print("Min goals @ home",minnesota_score_home)
print("Van goals @ away",vancouver_score_away)
print("Min goals @ away",minnesota_score_away)
'''

def simulate(base_model, homeTeam, awayTeam, max_goals=8):
	home_goals_avg = base_model.predict(pd.DataFrame
						(data={"Team" : homeTeam, "Opposition" : awayTeam, "home" : 'yes'},index=[1])).values[0]
	away_goals_avg = base_model.predict(pd.DataFrame(data={"Team" : awayTeam, "Opposition" : homeTeam, "home" : 'no'},index=[1])).values[0]
	
	# probability of scoring X goals given avg.
	pred = [[poisson.pmf(i, team_avg) for i in range(0,max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
	
	#print(np.array(pred[0])) # home team
	#print(np.array(pred[1])) # away team
	
	# P(H scores X -AND- A scores Y)
	return( np.outer(np.array(pred[0]), np.array(pred[1])) )

# P(Win|Canucks Home)
a = simulate(model, "Vancouver Canucks", "Minnesota Wild")
print("Canucks Home")
print("Canucks Win: ",np.sum(np.tril(a, k = -1)))
print("Overtime|Shootout: ", np.sum(np.diag(a, 0)))
print("Minnesota Win: ", np.sum(np.triu(a, k = 1)))
print("\n")


# P(Win|Canucks Away)
a = simulate(model, "Minnesota Wild", "Vancouver Canucks")
print("Canucks Away")
print("Minnesota Win: ",np.sum(np.tril(a, k = -1)))
print("Overtime|Shootout: ", np.sum(np.diag(a, 0)))
print("Canucks Win: ", np.sum(np.triu(a, k = 1)))
print("\n")

# P(Win|Boston Home)
a = simulate(model, "Boston Bruins", "Philadelphia Flyers")
print("Bos Home")
print("Bos Win: ",np.sum(np.tril(a, k = -1)))
print("Overtime|Shootout: ", np.sum(np.diag(a, 0)))
print("Phil Win: ", np.sum(np.triu(a, k = 1)))
print("\n")

# P(Win|Boston Away)
a = simulate(model, "Philadelphia Flyers", "Boston Bruins")
print("Bos Away")
print("Phil Win: ",np.sum(np.tril(a, k = -1)))
print("Overtime|Shootout: ", np.sum(np.diag(a, 0)))
print("Bos Win: ", np.sum(np.triu(a, k = 1)))
print("\n")

# P(Win|Pitt Home)
a = simulate(model, "Pittsburgh Penguins", "Montreal Canadiens")
print("Pitt Home")
print("Pitt Win: ",np.sum(np.tril(a, k = -1)))
print("Overtime|Shootout: ", np.sum(np.diag(a, 0)))
print("Mont Win: ", np.sum(np.triu(a, k = 1)))
print("\n")

# P(Win|Pitt Away)
a = simulate(model, "Montreal Canadiens", "Pittsburgh Penguins")
print("Pitt Away")
print("Mont Win: ",np.sum(np.tril(a, k = -1)))
print("Overtime|Shootout: ", np.sum(np.diag(a, 0)))
print("Pitt Win: ", np.sum(np.triu(a, k = 1)))
print("\n")


# P(Win|Tor Home)
a = simulate(model, "Toronto Maple Leafs", "Columbus Blue Jackets")
print("Tor Home")
print("Tor Win: ",np.sum(np.tril(a, k = -1)))
print("Overtime|Shootout: ", np.sum(np.diag(a, 0)))
print("Cbj Win: ", np.sum(np.triu(a, k = 1)))
print("\n")

# P(Win|Tor Away)
a = simulate(model, "Columbus Blue Jackets", "Toronto Maple Leafs")
print("Tor Away")
print("Cbj Win: ",np.sum(np.tril(a, k = -1)))
print("Overtime|Shootout: ", np.sum(np.diag(a, 0)))
print("Tor Win: ", np.sum(np.triu(a, k = 1)))
print("\n")


# P(Win|Col Home)
a = simulate(model, "Colorado Avalanche", "St. Louis Blues")
print("Col Win: ",np.sum(np.tril(a, k = -1)))
print("Overtime|Shootout: ", np.sum(np.diag(a, 0)))
print("StL Win: ", np.sum(np.triu(a, k = 1)))
print("\n")

# P(Win|Col Away)
a = simulate(model, "St. Louis Blues", "Colorado Avalanche")
print("StL Win: ",np.sum(np.tril(a, k = -1)))
print("Overtime|Shootout: ", np.sum(np.diag(a, 0)))
print("Col Win: ", np.sum(np.triu(a, k = 1)))
print("\n")

# P(Win|NYR Home)
a = simulate(model, "New York Rangers", "Carolina Hurricanes")
print("NYR Win: ",np.sum(np.tril(a, k = -1)))
print("Overtime|Shootout: ", np.sum(np.diag(a, 0)))
print("Car Win: ", np.sum(np.triu(a, k = 1)))
print("\n")

# P(Win|NYR Away)
a = simulate(model, "Carolina Hurricanes", "New York Rangers")
print("Car Win: ",np.sum(np.tril(a, k = -1)))
print("Overtime|Shootout: ", np.sum(np.diag(a, 0)))
print("NYR Win: ", np.sum(np.triu(a, k = 1)))
print("\n")

# P(Win|Chicago Home)
a = simulate(model, "Chicago Blackhawks", "Edmonton Oilers")
print("Chic Win: ",np.sum(np.tril(a, k = -1)))
print("Overtime|Shootout: ", np.sum(np.diag(a, 0)))
print("Edmt Win: ", np.sum(np.triu(a, k = 1)))
print("\n")

# P(Win|Chicago Away)
a = simulate(model, "Edmonton Oilers", "Chicago Blackhawks")
print("Edmt Win: ",np.sum(np.tril(a, k = -1)))
print("Overtime|Shootout: ", np.sum(np.diag(a, 0)))
print("Chic Win: ", np.sum(np.triu(a, k = 1)))
print("\n")

