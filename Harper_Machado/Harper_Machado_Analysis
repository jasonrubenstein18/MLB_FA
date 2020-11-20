# Stripped Data for Washington Nationals, Bryce Harper, Baltimore Orioles, and Manny Machado off of ESPN and Baseball Reference
# Idea of the study is to predict team attendance based on whether Harper/Machado is playing along with other regressors to isolate the Harper/Machado variable
# Take Log of dependent variable (attendance)
library(readr)
library(stats)
library(Hmisc)
library(arm)
library(ggplot2)
library(stargazer)
Harper.data <- read_csv("...")
Machado.data <- read_csv("...")

# Note that the Orioles home game on April 29, 2015 is removed from dataset due to riots that forced team to play game with no fans present
# Orioles already had 3 home games rescheduled that year so that will leave them with 239 across the 3 seasons as opposed to a full 243 game slate
# Convert binomial variables to numeric and Rank variable to factor
Harper.data$Play <- as.factor(Harper.data$Play)
Harper.data$`D/N` <- as.numeric(Harper.data$`D/N`)
Harper.data$Rank <- as.factor(Harper.data$Rank)
Harper.data$Attendance <- as.numeric(Harper.data$Attendance)
Harper.data$Month <- as.factor(Harper.data$Month)
#View(Harper.data)

Machado.data$Play <- as.factor(Machado.data$Play)
Machado.data$`D/N` <- as.numeric(Machado.data$`D/N`)
Machado.data$Rank <- as.factor(Machado.data$Rank)
Machado.data$Attendance <- as.numeric(Machado.data$Attendance)
Machado.data$Month <- as.factor(Machado.data$Month)

# We want to isolate the influence of Harper and Machado on their respective team's attendnace numbers
# To do this we have created a binomial value under the column name Play where 1 = Played and 0 = Didn't Play
# Reassign Play variable to binomial

# Simple regression on Play variable and Attendance
Harper.attendance <- Harper.data$Attendance
Harper.play <- Harper.data$Play
Harper.month <- Harper.data$Month
Machado.attendance <- Machado.data$Attendance
Machado.play <- Machado.data$Play
Machado.month <- Machado.data$Month

Harper.Simple <- data.frame(Harper.attendance, Harper.play)
Machado.Simple <- data.frame(Machado.attendance, Machado.play)
summary(Harper.Simple)
summary(Machado.Simple)
SimpleModel <- lm(Harper.data$Attendance ~ Harper.data$Play)
summary(SimpleModel)


# This produced highly biased results, there is certainly omitted variable bias present
# Choose to introduce variables of opponent, Day/Night (1 = Night, 0 = Day), Rank (within division standings), and Year based effect. This influence will be noticeable as "hype" entering season has some influence on attendance
# I consider adding in statistics to quantify impact of Machado and Harper performance on team attendance, but elect not to due to sample size
Harper.rank <- Harper.data$Rank
Harper.time.of.game <- Harper.data$`D/N`
Harper.opponent <- Harper.data$Opp
Harper.year <- Harper.data$Year
Harper.Simple.Full <- data.frame(Harper.Simple, Harper.month, Harper.rank, Harper.time.of.game, Harper.opponent, Harper.year)

Machado.rank <- Machado.data$Rank
Machado.time.of.game <- Machado.data$`D/N`
Machado.opponent <- Machado.data$Opp
Machado.year <- Machado.data$Year
Machado.Simple.Full <- data.frame(Machado.Simple, Machado.month, Machado.rank, Machado.time.of.game, Machado.opponent, Machado.year)


# Running simple regression on impact of Harper/Machado playing to team attendance shows that Harper drew far more fans to games than Machado
Harper.Model.1 <- lm(Harper.attendance ~ Harper.play)
Harper.Model.1

Machado.Model.1 <- lm(Machado.attendance ~ Machado.play)
Machado.Model.1
# We see here initially that Harper is a far more impactful in drawing fans than Machado
# Consider more variables (Opponent, day/night time of game, division rank, and year)
# First we will consider division rank
Harper.Model.2 <- glm(Harper.attendance ~ Harper.play + Harper.rank)
Harper.Model.2

Machado.Model.2 <- glm(Machado.attendance ~ Machado.play + Machado.rank)
Machado.Model.2
# This added variable brings the two players together, but Harper remains a larger draw by about 350 fans

# Next we will consider day/night time of game (binomial; 0 = day, 1 = night)
Harper.Model.3 <- glm(Harper.attendance ~ Harper.play + Harper.rank + Harper.time.of.game)
Harper.Model.3
summary(Harper.Model.3)

Machado.Model.3 <- glm(Machado.attendance ~ Machado.play + Machado.rank + Machado.time.of.game)
Machado.Model.3
summary(Machado.Model.3)
# After adding in time of game variable, Harper remains far ahead of Machado. Also it should be noted that time of game is a statistically significant regressor of game attendance at 99% critical value


# Next, we would like to introduce Opponent as an underlying fixed effect in the regression
# Games in which teams host other top ranked teams (or rivals) attendance goes up. This variable is a necessary inclusion in the regression
Harper.Model.4 <- glm(Harper.attendance ~ Harper.play + Harper.rank + Harper.time.of.game + factor(Harper.opponent))
summary(Harper.Model.4)
confint(Harper.Model.4)

Machado.Model.4 <- glm(Machado.attendance ~ Machado.play + Machado.rank + Machado.time.of.game + factor(Machado.opponent))
Machado.Model.4
confint(Machado.Model.4)
# Now I explore the interaction between Play and Rank variables. It is certainly plausible that team would not force star player to play when ranked low in their divisons
Harper.Model.5 <- glm(Harper.attendance ~ Harper.play + Harper.rank + Harper.time.of.game + factor(Harper.opponent)
                      + paste(Harper.play, Harper.rank) 
                      + factor(Harper.month) - 1
                      )
summary(Harper.Model.5)
confint(Harper.Model.5)

Machado.Model.5 <- glm(Machado.attendance ~ Machado.play + Machado.rank + Machado.time.of.game +  factor(Machado.opponent) 
                      # + paste(Machado.rank, Machado.play)
                      + factor(Machado.month) - 1
                      )
summary(Machado.Model.5)
confint(Machado.Model.5)
# It is clear from this model that Harper is more valuable to the Nationals in terms of attendance than Machado is to the Orioles. From that, we can conclude that Harper would have a greater impact on team attendance than Machado


# Printing Model Results
Model.2.comparison <- stargazer(Harper.Model.2, Machado.Model.2, type = "text",
                                dep.var.labels = c("Nationals Attendance", "Orioles Attendance"),
                                covariate.labels = c("Play", "Rank"),
                                out = "Model2Harper.Machado.txt")

Model.3.comparison <- stargazer(Harper.Model.3, Machado.Model.3, type = "text",
                                dep.var.labels = c("Nationals Attendance", "Orioles Attendance"),
                                covariate.labels = c("Play", "Rank", "Time Of Game"),
                                out = "Model3Harper.Machado.txt")

Model.4.comparison <- stargazer(Harper.Model.4, Machado.Model.4, type = "text",
                                dep.var.labels = c("Nationals Attendance", "Orioles Attendance"),
                                covariate.labels = c("Play", "Rank", "Time Of Game", "Opponent Fixed Effect"),
                                out = "Model4Harper.Machado.txt")

Model.5.comparison <- stargazer(Harper.Model.5, Machado.Model.5, type = "text",
                                dep.var.labels = c("Nationals Attendance", "Orioles Attendance"),
                                covariate.labels = c("Play", "Rank", "Time Of Game", "Opponent Fixed Effect", "Play and.Rank Interaction Term"),
                                out = "Model5Harper.Machado.txt")




Model.2.output <- capture.output(Model.2.comparison)
Model.2.output <- cat("Harper v Machado Model 2", Model.2.comparison, file = "Harper.Machado.Model.2.txt", sep = ",", append = TRUE)

Model.3.output <- capture.output(Model.3.comparison)
Model.3.output <- cat("Harper v Machado Model 3", Model.3.comparison, file = "Harper.Machado.Model.3.txt", sep = ",", append = TRUE)

Model.4.output <- capture.output(Model.4.comparison)
Model.4.output <- cat("Harper v Machado Model 4", Model.4.comparison, file = "Harper.Machado.Model.4.txt", sep = ",", append = TRUE)

Model.5.output <- capture.output(Model.5.comparison)
Model.5.output <- cat("Harper v Machado Model 5", Model.5.comparison, file = "Harper.Machado.Model.5.txt", sep = ",", append = TRUE)

########

plot(Harper.play, Harper.attendance)
plot(Machado.play, Machado.attendance)

SimpleAttendanceHarper <- lm(Harper.attendance ~ Harper.play)
SimpleAttendanceMachado <- lm(Machado.attendance ~ Machado.play)
summary(SimpleAttendanceHarper)
summary(SimpleAttendanceMachado)

WARtoContractPlot <- plotly::plot_ly(data = FA.Contract.Factors, y = Contract.NPV, x = War1,  type = "scatter", mode = "markers", color = Name, opacity = .5)
WARtoContractPlot

AIC(SimpleContract) # Akaike Information Criterion for model selection


#OLS Regression for Predicting Contract NPV

ols <- lm(Contract.NPV ~ War1 + War2 + Age.At.FA, data = FA.Contract.Factors)
summary(ols)
yhatols <- ols$fitted.values
AIC(ols)
par(mar = rep(2,4))

#	Ordinary Least Squares Regression Plot for variables in predicting Contract NPV
plot(Contract.NPV, yhatols, pch = 100, xlab = "Contract NPV", ylab = "yhatols")
abline(lm(yhatols~Contract.NPV), lwd = 3, col = "red")

#	Introduce Positional Based Fixed Effects **
fixed.dum <- lm(Contract.NPV ~ War1 + War2 + Age.At.FA + factor(Position) - 1, data = FA.Contract.Factors)
summary(fixed.dum)
AIC(fixed.dum)

#	Predicting Values of xFIP- with Team Based Fixed Effect
yhatdum <- fixed.dum$fitted.values
summary(yhatdum)

#	Simple Plot and Predicted Examples
list(yhatdum[Name == "Yoenis Cespedes"])
list(yhatdum[Name == "Mark Trumbo"])
list(yhatdum[Name == "Mark Melancon"])
list(yhatdum[Name == "Aroldis Chapman"])
list(yhatdum[Name == "David Price"])

#	Regress Line through Predicted Values Similar to OLS
scatterplot(yhatdum~Contract.NPV, boxplots = FALSE, xlab = "Contract NPV", ylab = "yhat", smooth = FALSE, legend.plot = FALSE)
abline(lm(yhatdum~Contract.NPV), lwd = 3, col = "red") # Presence of heteroskedasticity possible

#	Interactive Plot of Predicted Values
NPVPredictPlot <- plotly::plot_ly(data = FA.Contract.Factors, y = yhatdum, x = Contract.NPV, type = "scatter", mode = "markers", color = Name, opacity = .5)
NPVPredictPlot


#	Display Table Comparing 2 Models in Latex Form
apsrtable::apsrtable(ols,fixed.dum, model.names = c("OLS", "OLS_DUM"))
anova(ols)
anova(fixed.dum)
#	Determine Position Based Fixed Effects
fixedteam <- plm(Contract.NPV ~ War1 + War2 + Age.At.FA, data = FA.Contract.Factors, index = c("Position"), model = "within")
summary(fixedteam)


fixef(fixedteam) # 	Display the fixed effects (constants for each position)
pFtest(fixedteam, ols) # Testing for fixed effects, null: OLS better than fixed
#	P-value < .05 so accept null hypothesis

#	Random Effects model using plm package
randomteam <- plm(Contract.NPV ~ War1 + War2 + Age.At.FA, data = FA.Contract.Factors, index=c("Position"), model = "random")
summary(randomteam) #	Coefficient is average effect of X over Y when X changes

#	Fixed vs. Random
phtest(fixedteam, randomteam) # P-value is less than .05 so use fixed effects

# Test for Heteroskedasticity
bptest(Contract.NPV ~  War1 + War2 + Age.At.FA + factor(Position), data = FA.Contract.Factors, studentize = FALSE)
# No heteroskedasticity present which is good

# Team Effects Model is Validated and reprinted
plotly::plot_ly(data = FA.Contract.Factors, y = yhatdum, x = Contract.NPV, type = "scatter", mode = "markers", color = Name, opacity = .5)
abline(plot(War1, Contract.NPV, xlab = "Most Recent Year WAR", ylab = "Contract NPV", main = "Relationship Between WAR and Contract NPV"))

#	Fixed effects specification with Akaike Information Criterion
anova(fixed.dum, fixedteam, test = "F")
AIC(SimpleContract)
AIC(fixed.dum) # Fixed-effect dummy variable more than doubled reliability of model
