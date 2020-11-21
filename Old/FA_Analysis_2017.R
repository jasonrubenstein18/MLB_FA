library(readr)
library(MASS)
library(stats)
library(foreign)
library(moments)
library(apsrtable)
library(car)
library(plm)
library(lmtest)
library(lme4)
FA.Valuation <- read_csv("...",
                         col_types = cols(`$(NPV) Per WAR` = col_number(), 
                                          `10% Investment Return y0` = col_number(), 
                                          `Age at FA` = col_number(), `Average WAR/Year` = col_number(), 
                                          `Contract Y0` = col_number(), `Contract Y1 (1 year after base)` = col_number(), 
                                          `Contract Y2` = col_number(), `Contract Y3` = col_number(), 
                                          `Contract Y4` = col_number(), `Contract Y5` = col_number(), 
                                          `Contract Y6` = col_number(), `Contract Y7` = col_number(), 
                                          `Most Likely Investment Return y1` = col_number(), 
                                          `Most Likely Investment Return y2` = col_number(), 
                                          `Most Likely Investment Return y3` = col_number(), 
                                          `Most Likely Investment Return y4` = col_number(), 
                                          `Most Likely Investment Return y5` = col_number(), 
                                          `Most Likely Investment Return y6` = col_number(), 
                                          `Most Likely Investment Return y7` = col_number(), 
                                          `NPV of Contract (2.7% U.S. Inflation Rate)` = col_number(), 
                                          `NPV of Contract with Most Likely Investment Return (10%)` = col_number(), 
                                          `NPV of Contract with Optimistic Investment Return (12%)` = col_number(), 
                                          `NPV of Contract with Pessimistic Investment Return (8%)` = col_number(), 
                                          `Optimistic Investment Return y0` = col_number(), 
                                          `Optimistic Investment Return y1` = col_number(), 
                                          `Optimistic Investment Return y2` = col_number(), 
                                          `Optimistic Investment Return y3` = col_number(), 
                                          `Optimistic Investment Return y4` = col_number(), 
                                          `Optimistic Investment Return y5` = col_number(), 
                                          `Optimistic Investment Return y6` = col_number(), 
                                          `Optimistic Investment Return y7` = col_number(), 
                                          `Pessimistic Investment Return y0` = col_number(), 
                                          `Pessimistic Investment Return y1` = col_number(), 
                                          `Pessimistic Investment Return y2` = col_number(), 
                                          `Pessimistic Investment Return y3` = col_number(), 
                                          `Pessimistic Investment Return y4` = col_number(), 
                                          `Pessimistic Investment Return y5` = col_number(), 
                                          `Pessimistic Investment Return y6` = col_number(), 
                                          `Pessimistic Investment Return y7` = col_number(), 
                                          `Total WAR last 6 years` = col_number(), 
                                          `War1 (most recent)` = col_number(), 
                                          War2 = col_number(), War3 = col_number(), 
                                          War4 = col_number(), War5 = col_number(), 
                                          War6 = col_number()))

#View(FA.Valuation)
FA.Valuation$Position <- as.character(FA.Valuation$Position)
str(FA.Valuation, 3)
FA.Valuation$`Qualifying Offer?` <- as.factor(FA.Valuation$`Qualifying Offer?`)


Name <- FA.Valuation$Player
Position <- FA.Valuation$Position
Qualifying.Offer <- FA.Valuation$`Qualifying Offer?`
War1 <- FA.Valuation$`War1 (most recent)`
War2 <- FA.Valuation$War2
War3 <- FA.Valuation$War3
War4 <- FA.Valuation$War4
War5 <- FA.Valuation$War5
War6 <- FA.Valuation$War6
Six.Year.War.Total <- FA.Valuation$`Total WAR last 6 years`
Age.At.FA <- FA.Valuation$`Age at FA`
Contract.NPV <- FA.Valuation$`NPV of Contract (2.7% U.S. Inflation Rate)`
Contract.NPV.with.Optimistic.Investment <- FA.Valuation$`NPV of Contract with Optimistic Investment Return (12%)`
Contract.NPV.with.Pessimistic.Investment <- FA.Valuation$`NPV of Contract with Pessimistic Investment Return (8%)`
Contract.NPV.with.Likely.Investment <- FA.Valuation$`NPV of Contract with Most Likely Investment Return (10%)`
NPV.Per.WAR <- FA.Valuation$`$(NPV) Per WAR`
Agency <- FA.Valuation$Agency
Agent.Name <- FA.Valuation$Agent
NPV.Per.Year <- FA.Valuation$NPVperYear

# Squares and logs of variables
Age.At.FA.sq <- FA.Valuation$`Age at FA`^2
War1.sq <- FA.Valuation$`War1 (most recent)`^2
War2.sq <- FA.Valuation$War2^2
War3.sq <- FA.Valuation$War3^2
War4.sq <- FA.Valuation$War4^2
War5.sq <- FA.Valuation$War5^2
War6.sq <- FA.Valuation$War6^2
Six.Year.War.Total.sq <- FA.Valuation$`Total WAR last 6 years`^2

Age.At.FA.log <- log(FA.Valuation$`Age at FA`)
War1.log <- log(FA.Valuation$`War1 (most recent)`)
War2.log <- log(FA.Valuation$War2)
War3.log <- log(FA.Valuation$War3)
War4.log <- log(FA.Valuation$War4)
War5.log <- log(FA.Valuation$War5)
War6.log <- log(FA.Valuation$War6)
Six.Year.War.Total.log <- log(FA.Valuation$`Total WAR last 6 years`)



FA.Data <- data.frame(Name, Position, War1, War2, 
                      War3, War4, War5, War6,
                      War1.sq, War2.sq, 
                      War3.sq, War4.sq, War5.sq, War6.sq,
                      War1.log, War2.log, 
                      War3.log, War4.log, War5.log, War6.log,
                      Age.At.FA, Contract.NPV, Six.Year.War.Total,
                      Agency, Agent.Name, Contract.NPV.with.Likely.Investment,
                      Contract.NPV.with.Optimistic.Investment, Contract.NPV.with.Pessimistic.Investment,
                      Contract.NPV, NPV.Per.Year, Qualifying.Offer)
hist(War1.sq)
hist(War2.sq)
hist(Six.Year.War.Total.sq)

# Split into train/test sets
set.seed(69)
train_ind <- sample(seq_len(nrow(FA.Data)), size = 86)

train <- FA.Data[train_ind, ]
test <- FA.Data[-train_ind, ]
test$predicted_price <- NA

SP <- subset(FA.Data, Position == "1")
RP <- subset(FA.Data, Position == "2")
Catcher <- subset(FA.Data, Position == "3")
Firstbase <- subset(FA.Data, Position == "4")
Secondbase <- subset(FA.Data, Position == "5")
Shortstop <- subset(FA.Data, Position == "6")
Thirdbase <- subset(FA.Data, Position == "7")
CornerOF <- subset(FA.Data, Position == "8")
CenterOF <- subset(FA.Data, Position == "9")


###### Scrapwork #####
# Linear Model to predict Contract NPV at 2.7% interest rate
# Regressors of WAR in each of prior six years and age at time of FA
FA.Contract.Model.Six.Year.War.Total <- glm(data = train, Contract.NPV ~ Six.Year.War.Total)
summary(FA.Contract.Model.Six.Year.War.Total)

test$predicted_price <- predict(FA.Contract.Model.Six.Year.War.Total, newdata=test, allow.new.levels = TRUE)
plot(test$Contract.NPV, test$predicted_price)

New.Model1 <- glm(Contract.NPV ~ Six.Year.War.Total + Six.Year.War.Total.sq + Six.Year.War.Total.log)
summary(New.Model1)
# Single regressor is statistically significant at 99% level

# Introduce most recent year before FA WAR
FA.Contract.Model.1 <- glm(Contract.NPV ~ Six.Year.War.Total + War1)
summary(FA.Contract.Model.1)



New.Model2 <- glm(Contract.NPV ~ Six.Year.War.Total.sq + War1 + War1.sq)
summary(New.Model2)

New.Model3 <- glm(Contract.NPV ~ Six.Year.War.Total.sq + War1.sq)
# The square of the variable here is more accurate in predicting Contract NPV

# Inroduce 2nd most recent year before FA WAR
FA.Contract.Model.2 <- glm(Contract.NPV ~ Six.Year.War.Total + War1.sq + War2)
summary(FA.Contract.Model.2)

New.Model3 <- glm(Contract.NPV ~ Six.Year.War.Total.sq + War1.sq + War2.sq)
summary(New.Model3)
# 6-year WAR total is no longer statistically significant. Just the most recent 2 years are at a 99% level though

# Introduce 3rd most recent year before FA WAR
FA.Contract.Model.3 <- glm(Contract.NPV ~ Six.Year.War.Total + War1 + War2 + War3)
summary(FA.Contract.Model.3)
# 3rd most recent year before FA is not statistically significant to predicting FA contract NPV. We can determine from this that teams consider just the most recent two years

# Introduce positional based fixed effects where (1 = SP, 2 = RP, 3 = C, 4 = 1B, 5 = 2B, 6 = SS, 7 = 3B, 8 = Corner OF, 9 = CF)
FA.Contract.Model.4 <- glm(Contract.NPV ~ Six.Year.War.Total + War1 + War2
                           + factor(Position))
summary(FA.Contract.Model.4)

test$predicted_price <- predict(FA.Contract.Model.4, newdata=test, allow.new.levels = TRUE)
plot(test$Contract.NPV, test$predicted_price)
# We see here that in 2015-2016 pitchers were granted large salary markups when compared to position players with similar WAR in the two most recent years
# This means that, with equal WAR scores, pitchers will be paid significantly more than hitters

# Now I will introduce the fixed effect that age has on contracts in free agency. Teams want to sign players in their primes so they may be more willing to sign greater NPV deals to younger players
FA.Contract.Model.5 <- glm(Contract.NPV ~ Six.Year.War.Total.sq + War1.sq + War2.sq
                           + factor(Age.At.FA) + factor(Position))
summary(FA.Contract.Model.5)

# Now I will introduce an interaction term between the Position and Age to see if teams view age as less important depending on position
FA.Contract.Model.6 <- glm(Contract.NPV ~ Six.Year.War.Total.sq + War1.sq + War2.sq
                           + factor(Age.At.FA) + paste(Position, Age.At.FA))
summary(FA.Contract.Model.6)
# It is seen here that 29 and 30 year old produces most statistically significant data points. This makes sense intuitively because FA data is from players in that range

# Examine impact of Age and most recent by positions
Mixed_Model <- lmer(Contract.NPV ~ War1.sq + War2.sq + Qualifying.Offer + Age.At.FA
					  + (1+Age.At.FA|Position) + (1+War1.sq|Position), data = train)
summary(Mixed_Model)
coef(Mixed_Model)
test$predicted_price <- predict(Mixed_Model, newdata=test, allow.new.levels = TRUE)
plot(test$Contract.NPV, test$predicted_price)

#Introduce effect of agents and agency given position and age (Remove 6 year WAR total because it is not statistically significant)
Agent.Effect.Model.1 <- glm(Contract.NPV ~ Six.Year.War.Total.sq + War1.sq + War2.sq
                            + Age.At.FA
                          # + factor(Agent.Name)
                          #  + factor(Position)
                            + paste(Agent.Name, Position)
                            )
summary(Agent.Effect.Model.1)
mean(Agent.Effect.Model.1$fitted.values[Agent.Name=="Scott Boras"], na.rm = TRUE)
# We see from this regression that Scott Boras has derived the most value for starting pitchers and first basemen on the open market
# Missing a lot of agent data so choose to exclude that data point



# The difference between open market NPV and actual NPV can be attributed to simple microeconomics. When there is a high demand for limited supply the price will be driven up. This will certainly be the case with Britton as he is a fantastic player who every team will surely like to have if he reaches free agency.



# I argue that there is survivor bias present when looking at impact of Age on Contract NPV
hist(Age.At.FA)
plot(density(Age.At.FA))
# We see a high number of observations between the ages of 30 and 33 and far fewer at higher ages
# Players that are good enough to continue to receive contracts to their late 40s must be good as team's generally prefer to sign younger players that produce similarly to older ones
summary(Age.At.FA)

# Find how player position impacts team's $ spent in NPV per WAR
# Start by regressing NPV per WAR by WAR in most recent season
NPV.Model.1 <- glm(NPV.Per.WAR ~ War1)
summary(NPV.Model.1)

# Introduce second most recent season WAR
NPV.Model.2 <- glm(NPV.Per.WAR ~ War1 + War2)
summary(NPV.Model.2)
# See that second most recent season is not statistically significant in predcting $ spent in NPV per WAR

# Now include the positional fixed effect
NPV.Model.3 <- glm(NPV.Per.WAR ~ War1 + factor(Position) - 1)
summary(NPV.Model.3)
# Here we see that pitchers experience a markup in $ per WAR produced. This inherently makes sense given that I found earlier that Pitchers are generally given a greater NPV contract for similar win production compared to batters
# Next I implement the agency fixed effect to see which agencies are deriving most value for clients
NPV.Model.4 <- glm(NPV.Per.WAR ~ War1 + paste(Position, Agency))
summary(NPV.Model.4)
# From this we see that Boras Corp has derived near top value in terms of NPV per WAR produced for each player position
# Now I will evaluate the relatonship of NPV to individual agents, (Assuming that Boras Corp agent is Scott Boras)
NPV.Model.5 <- glm(NPV.Per.WAR ~ War1 + factor(Position) + factor(Agent.Name) - 1)
summary(NPV.Model.5)
# We see here that Scott Boras has derived top notch value for his clients


# Finally given the models that I have created I will input the seven players to the regression to predict career future earnings
# Here is the work for projecting prospect earnings given 6-year WAR projection
Prospect.Model.1 <- glm(Contract.NPV ~ Six.Year.War.Total)
summary(Prospect.Model.1)

# Factor in positional effect
Prospect.Model.2 <- glm(Contract.NPV ~ Six.Year.War.Total + factor(Position))
summary(Prospect.Model.2)

# Now we will factor age into the regression
Prospect.Model.3 <- glm(Contract.NPV ~ Six.Year.War.Total + Age.At.FA + factor(Position))
summary(Prospect.Model.3)

# Introduce interaction term between agent and player
Prospect.Model.4 <- glm(Contract.NPV ~ Six.Year.War.Total + Age.At.FA + paste(Agency, Position))
summary(Prospect.Model.4)

# Final Model 5
Prospect.Model.5 <- glm(Contract.NPV ~ Six.Year.War.Total + Age.At.FA + factor(Position) + paste(Agency, Position))
summary(Prospect.Model.5)





###### Model to determine contract NPV #####
plot(War1.sq, Contract.NPV) # Square of WAR1 to NPV plot appears fit linearly

SimpleContract <- lm(Contract.NPV ~ War1.sq)
summary(SimpleContract)
WARtoContractPlot <- plotly::plot_ly(data = FA.Data, y = Contract.NPV, x = War1.sq,  type = "scatter", mode = "markers", color = Name, opacity = .5)
WARtoContractPlot

AIC(SimpleContract) # Akaike Information Criterion for model selection


#OLS Regression for Predicting Contract NPV

ols <- lm(Contract.NPV ~ War1.sq + War2.sq + Age.At.FA + Six.Year.War.Total.sq)
summary(ols)
yhatols <- ols$fitted.values
AIC(ols)
par(mar = rep(2,4))

#	Ordinary Least Squares Regression Plot for variables in predicting Contract NPV
plot(Contract.NPV, yhatols, pch = 25, xlab = "Contract NPV", ylab = "yhatols")
abline(lm(yhatols~Contract.NPV), lwd = 3, col = "red")

#	Introduce Positional Based Fixed Effects **
fixed.dum <- lm(Contract.NPV ~ War1.sq + War2.sq + Age.At.FA + Six.Year.War.Total.sq + factor(Position) - 1)
summary(fixed.dum)
AIC(fixed.dum)

54.76672 + 2.64510*2.4^2 + 1.24722*3.8^2-1.49409*32+.04081*20.2

#	Predicting Values of xFIP- with Team Based Fixed Effect
yhatdum <- fixed.dum$fitted.values
summary(yhatdum)

#	Simple Plot and Predicted Examples
list(yhatdum[Name == "Yoenis Cespedes"])
list(yhatdum[Name == "Mark Trumbo"])
list(yhatdum[Name == "Mark Melancon"])
list(yhatdum[Name == "Aroldis Chapman"]) # Model undervalues Relief Pitchers. Theorize that they are not properly valued by WAR stat
list(yhatdum[Name == "David Price"])

#	Regress Line through Predicted Values Similar to OLS
scatterplot(yhatdum~Contract.NPV, boxplots = FALSE, xlab = "Contract NPV", ylab = "yhat", smooth = FALSE, legend.plot = FALSE)
abline(lm(yhatdum~Contract.NPV), lwd = 3, col = "red") # Presence of heteroskedasticity possible

#	Interactive Plot of Predicted Values
NPVPredictPlot <- plotly::plot_ly(y = Contract.NPV, x = yhatdum, type = "scatter", mode = "markers", color = Name, opacity = .5)
NPVPredictPlot


#	Display Table Comparing 2 Models in Latex Form
apsrtable::apsrtable(ols,fixed.dum, model.names = c("OLS", "OLS_DUM"))
anova(ols)
anova(fixed.dum)
#	Determine Position Based Fixed Effects
fixedpos <- plm(data = FA.Data, Contract.NPV ~ War1.sq + War2.sq + Age.At.FA, index = c("Position"), model = "within")
summary(fixedpos)


fixef(fixedpos) # 	Display the fixed effects (constants for each position)
pFtest(fixedpos, ols) # Testing for fixed effects, null: OLS better than fixed
#	P-value < .05 so accept null hypothesis

#	Random Effects model using plm package
randompos <- plm(Contract.NPV ~ War1.sq + War2.sq + Age.At.FA, data = FA.Data, index=c("Position"), model = "random")
summary(randompos) #	Coefficient is average effect of X over Y when X changes

 #	Fixed vs. Random
phtest(fixedpos, randompos) # P-value is less than .05 so use fixed effects

# Test for Heteroskedasticity
bptest(Contract.NPV ~  War1.sq + War2.sq + Age.At.FA + factor(Position), data = FA.Data, studentize = FALSE)
# No heteroskedasticity present which is good

# Position Effects Model is Validated and reprinted
plotly::plot_ly(data = FA.Data, y = Contract.NPV, x = yhatdum, type = "scatter", mode = "markers", color = Name, opacity = .5)
abline(plot(War1.sq, Contract.NPV, xlab = "Most Recent Year WAR", ylab = "Contract NPV", main = "Relationship Between WAR and Contract NPV"))

#	Fixed effects specification with Akaike Information Criterion
anova(fixed.dum, fixedpos, test = "F")
AIC(SimpleContract)
AIC(fixed.dum) # Fixed-effect dummy variable improves reliability of model



# Now look at just positional effect again
Full.Model.1 <- glm(Contract.NPV ~ Six.Year.War.Total.sq
                    + War1.sq + War2.sq 
                    + Age.At.FA #+ factor(Qualifying.Offer) - 1
                    #+ log(Age.At.FA) # Positional fixed effect coefficients is most accurate with log of age OR non-squared value
                    # I argue that log of Age is most accurate because that is the expected relationship for age to contract value
                    # At certain age, if you are still good enough to receive a contract, additional years of age are meaningless
                    + factor(Position) - 1
                    , data = FA.Data
                    )
summary(Full.Model.1)


Full.Model.2 <- glm(Contract.NPV ~ Six.Year.War.Total.sq 
                    + War1.sq + War2.sq #+ Qualifying.Offer
                    + log(Age.At.FA)
                    + paste(Agent.Name, Position)
                    )
summary(Full.Model.2)


# As for Britton I would work on another model because relief pitchers are undervalued in WAR statistic

# To develop contract proposal for Relief Pitchers must account for saves, K-rate, GB%, etc.

NPV.Per.Year.Model.1 <- lm(NPV.Per.Year ~ War1.sq + War1 #+ War2.sq #+ War3.sq + War4.sq + War5.sq + War6.sq
                           + Six.Year.War.Total.sq # + Age.At.FA 
                          # + Age.At.FA.log # Age also not signifanct influence on NPV per Year. We see this in many examples where players may receive large single year contracts at older age
                          # + Contract.NPV # Higher contract NPV correlates highly with large average NPV in life of contract
                          # + factor(Position) - 1  # See that position is not a significant influence on NPV per year
                          )
summary(NPV.Per.Year.Model.1)



plotly::plot_ly(data = FA.Data, y = Contract.NPV, x = War1.sq, type = "scatter", mode = "markers", color = Name, opacity = .5)
