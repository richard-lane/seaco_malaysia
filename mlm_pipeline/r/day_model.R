# install.packages("lme4", repos="https://www.stats.bris.ac.uk/R/")
# install.packages("sjPlot", repos="https://www.stats.bris.ac.uk/R/")
# install.packages("effects", repos="https://www.stats.bris.ac.uk/R/")
library(lme4)
library(tidyverse)
library(sjPlot)

# Read the data
model_df <- read_csv("mlm_pipeline/data/model_df.csv")

# Define some models
# random_intercept <- glmer(entry ~ 1 + (day|p_id), data=model_df, family = binomial(link = "logit"))
random_both <- glmer(entry ~ day + (1 + day|p_id), data = model_df, family = binomial(link = "logit"))

# Extract random effects
my_plot = plot_model(random_both, type = "eff", terms="day", show.rug=TRUE)
my_plot + scale_y_continuous(limits = c(0.0, 1.0), label=scales::percent_format(accuracy = 10))
ggsave("fit.png")

# Plot histograms
random_effects <- ranef(random_both)[1]$p_id
random_effects <- tidyr::pivot_longer(random_effects, everything())
ggplot(random_effects, aes(x=value)) +
    geom_histogram(binwidth=0.2) +
    facet_wrap(~name, scales="free") +
    labs(x="value", y="Count", title="Random Effects")

ggsave("histograms.png")

# Anova