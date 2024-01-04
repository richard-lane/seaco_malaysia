# install.packages("lme4", repos="https://www.stats.bris.ac.uk/R/")
# install.packages("sjPlot", repos="https://www.stats.bris.ac.uk/R/")
# install.packages("effects", repos="https://www.stats.bris.ac.uk/R/")
library(lme4)
library(tidyverse)
library(sjPlot)

# Read the data
model_df <- read_csv("mlm_pipeline/data/model_df.csv")

# Define some models
fixed_only <- glm(entry ~ day, data = model_df, family = binomial(link = "logit"))
random_intercept <- glmer(entry ~ day + (1 | p_id), data = model_df, family = binomial(link = "logit"))
random_both <- glmer(entry ~ day + (1 + day | p_id), data = model_df, family = binomial(link = "logit"))

fixed_plot <- plot_model(fixed_only, type = "eff", terms = "day", show.rug = TRUE)
fixed_plot + scale_y_continuous(limits = c(0.0, 1.0), label = scales::percent_format(accuracy = 10))
ggsave("fixed_only_fit.png")

intercept_plot <- plot_model(random_intercept, type = "eff", terms = "day", show.rug = TRUE)
intercept_plot + scale_y_continuous(limits = c(0.0, 1.0), label = scales::percent_format(accuracy = 10))
ggsave("random_intercepts.png")

full_plot <- plot_model(random_both, type = "eff", terms = "day", show.rug = TRUE)
full_plot + scale_y_continuous(limits = c(0.0, 1.0), label = scales::percent_format(accuracy = 10))
ggsave("random_pids_fit.png")

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
