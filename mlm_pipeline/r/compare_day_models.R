# install.packages("lme4", repos="https://www.stats.bris.ac.uk/R/")
# install.packages("sjPlot", repos="https://www.stats.bris.ac.uk/R/")
# install.packages("effects", repos="https://www.stats.bris.ac.uk/R/")
library(lme4)
library(tidyverse)
library(sjPlot)
library(dplyr)
library(scales)
library(ggplot2)

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
    plot <- plot_model(model, type = "pred", terms = "day", show.rug = TRUE)

    plot <- plot + geom_point(data = percentage_yes, aes(x = day, y = percentage_yes / 100), color = "#653D9BC4")
    plot <- plot + scale_y_continuous(limits = c(0.0, 1.0), label = percent_format(accuracy = 10))

    ggsave(filename, plot)
}

plot_and_save(fixed_only, "fixed_only_fit.png")
plot_and_save(random_intercept, "random_intercepts.png")
plot_and_save(random_both, "random_pids_fit.png")

plot_individual_pids <- function(model, filename) {
    # Create a new data frame with the unique days and p_ids
    newdata <- expand.grid(day = unique(model_df$day), p_id = unique(model_df$p_id))

    # Add the predicted values to the new data frame
    newdata$pred <- predict(model, newdata = newdata, type = "response")

    # Convert p_id to a factor
    newdata$p_id <- as.factor(newdata$p_id)

    # Create the overall plot
    plot <- plot_model(model, type = "pred", terms = "day", show.rug = TRUE)

    # Add the predicted values for each p_id to the plot
    plot <- plot + geom_line(data = newdata, aes(x = day, y = pred, color = p_id), alpha = 0.5) +
        scale_color_manual(values = rep("skyblue", 83)) +
        geom_point(data = percentage_yes, aes(x = day, y = percentage_yes / 100), color = "black") +
        scale_y_continuous(limits = c(0.0, 1.0), label = scales::percent_format(accuracy = 10)) +
        theme(legend.position = "none")

    ggsave(filename, plot)
}
plot_individual_pids(random_intercept, "random_intercepts_all.png")
plot_individual_pids(random_both, "random_pids_fit_all.png")


# Plot histograms
random_effects <- ranef(random_both)[1]$p_id
random_effects <- tidyr::pivot_longer(random_effects, everything())
ggplot(random_effects, aes(x = value)) +
    geom_histogram(binwidth = 0.2) +
    facet_wrap(~name, scales = "free") +
    labs(x = "value", y = "Count", title = "Random Effects")

ggsave("random_pid_effects_hists.png")

# Anova - to motivate having both a random intercept and slope
out_file <- "day_models.txt"
capture.output(summary(fixed_only), file = out_file)
capture.output(summary(random_intercept), file = out_file, append = TRUE)
capture.output(summary(random_both), file = out_file, append = TRUE)
capture.output(anova(random_intercept, random_both), file = out_file, append = TRUE)

aic_values <- data.frame(
    model = c("fixed_only", "random_intercept", "random_both"),
    AIC = c(AIC(fixed_only), AIC(random_intercept), AIC(random_both)),
    BIC = c(BIC(fixed_only), BIC(random_intercept), BIC(random_both))
)

capture.output(aic_values, file = out_file, append = TRUE)
