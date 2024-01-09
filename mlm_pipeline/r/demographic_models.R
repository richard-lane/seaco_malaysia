library(lme4)
library(tidyverse)
library(sjPlot)
library(dplyr)
library(scales)
library(ggplot2)
library(ggeffects)

# Read the data
model_df <- read_csv("mlm_pipeline/data/model_df.csv")

# Define some models
control <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
sex_model <- glmer(entry ~ day * sex + (1 + day | p_id), data = model_df, family = binomial(link = "logit"), control = control)
ethnicity_model <- glmer(entry ~ day * ethnicity + (1 + day | p_id), data = model_df, family = binomial(link = "logit"), control = control)
age_model <- glmer(entry ~ day * age_group + (1 + day | p_id), data = model_df, family = binomial(link = "logit"), control = control)
school_model <- glmer(entry ~ day * over_2_days_in_school + (1 + day | p_id), data = model_df, family = binomial(link = "logit"), control = control)
weekend_model <- glmer(entry ~ day * is_weekend + (1 + day | p_id), data = model_df, family = binomial(link = "logit"), control = control)

# Find what percentage of positive entries there were on each day
percentage_yes <- model_df %>%
    group_by(day) %>%
    summarise(percentage_yes = mean(entry) * 100)

plot_and_save <- function(model, covariate, filename) {
    sjp <- plot_model(model, type = "pred", terms = c("day", covariate))
    sjp_data <- sjp$data

    plot <- ggplot() +
        geom_line(data = sjp_data, aes(x = x, y = predicted, color = group)) +
        geom_ribbon(data = sjp_data, aes(x = x, ymin = conf.low, ymax = conf.high, fill = group), alpha = 0.1) +
        geom_point(data = percentage_yes, aes(x = day, y = percentage_yes / 100), color = "black") +
        scale_y_continuous(limits = c(0.0, 1.0), label = scales::percent_format(accuracy = 1)) +
        labs(color = covariate, fill = covariate)

    ggsave(filename, plot)
}

plot_and_save(sex_model, "sex", "mlm_pipeline/outputs/sex_fit.png")
plot_and_save(ethnicity_model, "ethnicity", "mlm_pipeline/outputs/ethnicity_fit.png")
plot_and_save(age_model, "age_group", "mlm_pipeline/outputs/age_fit.png")
plot_and_save(school_model, "over_2_days_in_school", "mlm_pipeline/outputs/school_fit.png")
plot_and_save(weekend_model, "is_weekend", "mlm_pipeline/outputs/weekend_fit.png")

capture.output(summary(sex_model), file = "sex_model.txt")
capture.output(summary(ethnicity_model), file = "ethnicity_model.txt")
capture.output(summary(age_model), file = "age_model.txt")
capture.output(summary(school_model), file = "school_model.txt")
capture.output(summary(weekend_model), file = "weekend_model.txt")

# Plot the model
plot_model(sex_model, type = "pred", terms = c("day", "sex"))
ggsave("tmp.png")
