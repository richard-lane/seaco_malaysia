library(lme4)
library(tidyverse)
library(sjPlot)
library(dplyr)
library(scales)
library(ggplot2)

# Read the data
model_df <- read_csv("mlm_pipeline/data/model_df.csv")

# Define some models
sex_model <- glmer(entry ~ day * sex + (1 + day | p_id), data = model_df, family = binomial(link = "logit"))
ethnicity_model <- glmer(entry ~ day * ethnicity + (1 + day | p_id), data = model_df, family = binomial(link = "logit"))
age_model <- glmer(entry ~ day * age + (1 + day | p_id), data = model_df, family = binomial(link = "logit"))

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

plot_and_save(sex_model, "sex_fit.png")
plot_and_save(ethnicity_model, "ethnicity_fit.png")
plot_and_save(age_model, "age_fit.png")

capture.output(summary(sex_model), file = "sex_model.txt")
capture.output(summary(ethnicity_model), file = "ethnicity_model.txt")
capture.output(summary(age_model), file = "age_model.txt")
