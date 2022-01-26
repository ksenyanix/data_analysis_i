## Quiz 12

# empty the working dictionary
rm(list = ls())
# set the working dictionary
setwd("~/Master UIBK/Semester 4/dataanalytics")
# import data
work <- read.csv("./data/work.csv", stringsAsFactors = TRUE)
summary(work)
head(work)

#Plots
plot(log(wage) ~ interaction(education, union), data = work)
with(work, interaction.plot(education, union, log(wage)))

#Anova
w1   <- lm(log(wage) ~ 1,                       data = work)
wE   <- lm(log(wage) ~ education,               data = work)
wU   <- lm(log(wage) ~ union,                   data = work)
wEU  <- lm(log(wage) ~ education + union,       data = work)
wExU <- lm(log(wage) ~ education * union,       data = work)

anova(w1, wE, wU, wEU, wExU)
# choose model 4
summary(wEU)


# Intercept = Intercept + ß1*0 + ß2*1/0
I_nounion  <- 6.848367 + 0.083525*0 + 0.038588*0
I_nounion
S_nounion  <- 0.083525
I_yesunion <- 6.848367 + 0.083525*0 + 0.038588*1
I_yesunion
s_yesunion <- 0.083525+0.038588*1
s_yesunion

# Why would we get here not the same results? 
summary(wExU)
I_nou  <- 6.882774 + 0.081154*0 -0.034287*0
I_nou
s_nou <- 0.081154
I_yesu <- 6.882774 + 0.081154*0 -0.034287*1
I_yesu
S_yesu <- 0.081154 - 0.034287 +0.005127
S_yesu
