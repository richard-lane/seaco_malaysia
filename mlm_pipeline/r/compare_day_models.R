# install.packages("lme4", repos="https://www.stats.bris.ac.uk/R/")
# install.packages("sjPlot", repos="https://www.stats.bris.ac.uk/R/")
# install.packages("effects", repos="https://www.stats.bris.ac.uk/R/")
library(lme4)
library(tidyverse)
library(sjPlot)
library(dplyr)
library(scales)

# Read the data
model_df <- read_csv("mlm_pipeline/data/model_df.csv")

# Define some models
fixed_only <- glm(entry ~ day, data = model_df, family = binomial(link = "logit"))
random_intercept <- glmer(entry ~ day + (1 | p_id), data = model_df, family = binomial(link = "logit"))
random_both <- glmer(entry ~ day + (1 + day | p_id), data = model_df, family = binomial(link = "logit"))

# Find what percentage of positive entries there were on each day
percentage_yes <- model_df %>%
    group_by(day) %>%
    summarise(percentage_yes = mean(entry) * 100)

plot_and_save <- function(model, filename) {
    plot <- plot_model(model, type = "eff", terms = "day", show.rug = TRUE)
    plot <- plot + scale_y_continuous(limits = c(0.0, 1.0), label = percent_format(accuracy = 10))
    ggsave(filename, plot)
}

plot_and_save(fixed_only, "fixed_only_fit.png")
plot_and_save(random_intercept, "random_intercepts.png")
plot_and_save(random_both, "random_pids_fit.png")


# Plot histograms
random_effects <- ranef(random_both)[1]$p_id
random_effects <- tidyr::pivot_longer(random_effects, everything())
ggplot(random_effects, aes(x = value)) +
    geom_histogram(binwidth = 0.2) +
    facet_wrap(~name, scales = "free") +
    labs(x = "value", y = "Count", title = "Random Effects")

ggsave("random_pid_effects_hists.png")

# Anova - to motivate having both a random intercept and slope
anova(random_intercept, random_both)

aic_values <- data.frame(
    model = c("fixed_only", "random_intercept", "random_both"),
    AIC = c(AIC(fixed_only), AIC(random_intercept), AIC(random_both)),
    BIC = c(BIC(fixed_only), BIC(random_intercept), BIC(random_both))
)

print(aic_values)
