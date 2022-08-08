## Using Stochastic Variational Inference to predict variable importance

### Introduction
  Variational inference methods are useful in scenarios where we need to compute the posterior distribution over nodes in a graphical model, but we cannot directly compute the posterior distribution.
This project utilizes the League of Legends Diamond Ranked Games (link) database from Kaggle, in which we try to model the importance of several parameters and evaluate the learned models by predicting final game outcomes.

  There are 39 columns in the data: 19 per team features and one “blueWins” outcome. Each team (blue and red) has the following features: 
AvgLevel, TotalExperience, ExperienceDiff,
WardsPlaced, WardsDestroyed,
CSPerMin, GoldPerMin, GoldDiff, TotalGold,
Deaths, FirstBlood, Assists, Kills,
Heralds, Dragons, TotalJungleMinionsKilled, EliteMonsters,
TotalMinionsKilled, and TowersDestroyed.
Among several features, there are game-specific resources that provide certain benefits. The table below is the Kaggle page’s glossary of game terms and definitions, which explains the significance of certain features.
Warding totem
An item that a player can put on the map to reveal the nearby area. Very useful for map/objectives control.
Minions
NPC that belong to both teams. They give gold when killed by players.
Jungle minions
NPC that belong to NO TEAM. They give gold and buffs when killed by players.
Elite monsters
Monsters with high hp/damage that give a massive bonus (gold/XP/stats) when killed by a team.
Dragons
Elite monster which gives team bonus when killed. The 4th dragon killed by a team gives a massive stats bonus. The 5th dragon (Elder Dragon) offers a huge advantage to the team.
Herald
Elite monster which gives stats bonus when killed by the player. It helps to push a lane and destroys structures.


### Resources
  We used Pyro SVI and modules such as numpy, pandas, and matplotlib to visualize and analyze our dataset and the results quantitatively. The model was trained on a local Jupyter notebook.
The code that we wrote defines a model and a guide for SVI. The model takes in all of the variables that are used to evaluate the game outcome. Each feature was also normalized for comparison. The guide has parameters that define weights for each variable, which define the importance of the corresponding feature. Some of the variables are also expected to have an optimal value, which is another parameter in the guide. Those that do not will have their optimal values set to zero, since negative optimal values do not make sense.
We adapted the player skill model for game outcomes from Homework 5, but used Pyro’s SVI engine from Homework 6 to learn the model.

### Evaluation
For each team in each game, we calculate a “score” per team, which is the sum of subscores for each individual variable. The variable’s subscore is the square of the difference between the true value and the optimal value. For variables without an optimal value, the zero optimal simplifies the formula to the variable’s weight multiplied by its squared value. The team that has the higher score is predicted to win the match.
Below are 4 models of various feature combinations to demonstrate feature importance to the game outcome. 

#### Model 1
This model contains every feature in the dataset. Figure 1.1 shows the weights and optimals, while the 0.73 accuracy in Figure 1.2 serves as a comparison to other models. The dense distribution in Figure 1.1 shows a blend of influential and insignificant features. Interestingly, there are also features with negative weights and/or optimals.


#### Model 2
This simplified model contains 3 features chosen randomly: WardsPlaced, TotalGold, and ExperienceDiff. Total gold was overwhelmingly significant due to its large weight. Its optimal ratio also went into the negative, thus increasing the square difference between the optimal value and the true value — further emphasizing the feature’s importance. However, the accuracy suffers a bit due to the model’s simplicity.

#### Model 3
This model adds two features to Model 2: EliteMonsters and TotalJungleMinionsKilled. The accuracy in this model is very close to Model 1. Most of these features are a good predictor of the game outcome, and a lot of the less impactful features that were included in Model 1 are missing here, which makes for a more stable model that is more accurate than Model 2.


#### Model 4
This model disables the optimal value parameters for most of the features, meaning that those optimals are set to 0. Forcing the optimal to 0 greatly affected ExperienceDiff, pushing its predicted importance above that of TotalGold’s. This greatly destabilized model, with more random-looking spikes in the accuracy chart. Interestingly, the model starts off a lot more accurate, compared to other models. However, the accuracy goes down significantly near the end, similar to Model 1.

### Conclusion
  The combination of importance sampling and SVI was generally effective at modeling the features that affected the game outcome. For League of Legends, in particular, we determined that TotalGold and ExperienceDiff were among the most critical factors in determining the outcome of the matches in this dataset. According to Model 1, the complete ordering, from highest importance to lowest, is GoldPerMin, GoldDiff, TotalExperience, TotalGold, ExperienceDiff, AvgLevel, CSPerMin, Dragons, Kills, EliteMonsters, TotalMinionsKilled, WardsPlaced, TowersDestroyed, TotalJungleMinionsKilled, FirstBlood, Assists, Heralds, WardsDestroyed, and Deaths.
Another observation of note is the presence of negative optimal values and the destabilization of models in their absences. While creating the models, we assumed that, since the values are normalized to [0,1], setting the optimal value to 0 would be equivalent to not having one. However, the models have shown this to not be the case, and it makes sense in retrospect — the negative optimals diminish the distances between data points. For example, 0.2 and 0.4 have the same distance as 0 and 0.2, but only half the distance as -0.2 and 0.2. Squaring the differences would thus create differences in the sense of distances.
We also observed that model 3 had similar accuracy to model 1 and concluded that the consistency was primarily due to the removal of less impactful factors, which further supports that TotalGold and ExperienceDiff are major features. So, in essence, using model 1 as a reference, we were able to identify potentially important predictors and focus subsequent models on those factors, which resulted in simpler yet stable models




