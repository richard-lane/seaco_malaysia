library(lme4)
library(tidyverse)
library(sjPlot)
library(dplyr)
library(scales)
library(ggplot2)
library(ggeffects)

plot_and_save <- function(model, covariate, filename, legend) {
    sjp <- plot_model(model, type = "pred", terms = c("day", covariate))
    sjp_data <- sjp$data

    # Use revalue to map the default labels to the new labels
    sjp_data$group_col <- as.character(sjp_data$group_col)
    sjp_data$group_col <- plyr::revalue(sjp_data$group_col, legend)
    sjp_data$group_col <- as.character(sjp_data$group_col)

    # Reorder factor levels if covariate is 'weekday'
    if (covariate == "weekday") {
        sjp_data$group_col <- factor(sjp_data$group_col, levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
    }

    plot <- ggplot() +
        geom_line(data = sjp_data, aes(x = x, y = predicted, color = group_col)) +
        geom_ribbon(data = sjp_data, aes(x = x, ymin = conf.low, ymax = conf.high, fill = group_col), alpha = 0.1) +
        geom_point(data = percentage_yes, aes(x = day, y = percentage_yes / 100), color = "black") +
        scale_y_continuous(limits = c(0.0, 1.0), label = scales::percent_format(accuracy = 1)) +
        labs(color = covariate, fill = covariate) +
        xlab("Day")

    ggsave(filename, plot)
}

# Read the data
model_df <- read_csv("mlm_pipeline/data/model_df.csv")

# Find what percentage of positive entries there were on each day
percentage_yes <- model_df %>%
    group_by(day) %>%
    summarise(percentage_yes = mean(entry) * 100)

# Model options
# Sometimes some of the models dont converge unless i increase the number of iterations
# Also bobyqa is faster than the default
control <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

# Define some models
sex_model <- glmer(entry ~ day * sex + (1 + day | p_id), data = model_df, family = binomial(link = "logit"), control = control)
plot_and_save(sex_model, "sex", "mlm_pipeline/outputs/sex_fit.png", list(`0` = "Male", `1` = "Female"))
capture.output(summary(sex_model), file = "sex_model.txt")
# capture.output(confint(sex_model, parm = "beta_")["sex", ], file = "sex_model.txt", append = TRUE)

ethnicity_model <- glmer(entry ~ day * ethnicity + (1 + day | p_id), data = model_df, family = binomial(link = "logit"), control = control)
plot_and_save(ethnicity_model, "ethnicity", "mlm_pipeline/outputs/ethnicity_fit.png", list(`1` = "Ethnicity 1", `2` = "Ethnicity 2", `3` = "Ethnicity 3"))
capture.output(summary(ethnicity_model), file = "ethnicity_model.txt")

age_model <- glmer(entry ~ day * age_group + (1 + day | p_id), data = model_df, family = binomial(link = "logit"), control = control)
plot_and_save(age_model, "age_group", "mlm_pipeline/outputs/age_fit.png", list(`0` = "7 - 12", `1` = "13 - 17"))
capture.output(summary(age_model), file = "age_model.txt")

school_model <- glmer(entry ~ day * over_2_days_in_school + (1 + day | p_id), data = model_df, family = binomial(link = "logit"), control = control)
plot_and_save(school_model, "over_2_days_in_school", "mlm_pipeline/outputs/school_fit.png", list(`0` = "0-2 days", `1` = ">2 days"))
capture.output(summary(school_model), file = "school_model.txt")

weekend_model <- glmer(entry ~ day * is_weekend + (1 + day | p_id), data = model_df, family = binomial(link = "logit"), control = control)
plot_and_save(weekend_model, "is_weekend", "mlm_pipeline/outputs/weekend_fit.png", list(`0` = "Weekday", `1` = "Weekend"))
capture.output(summary(weekend_model), file = "weekend_model.txt")

weekday_model <- glmer(entry ~ day * weekday + (1 + day | p_id), data = model_df, family = binomial(link = "logit"), control = control)
plot_and_save(
    weekday_model,
    "weekday",
    "mlm_pipeline/outputs/weekday_fit.png",
    list(`1` = "Monday", `2` = "Tuesday", `3` = "Wednesday", `4` = "Thursday", `5` = "Friday", `6` = "Saturday", `7` = "Sunday")
)
capture.output(summary(weekday_model), file = "weekday_model.txt")

ramadan_model <- glmer(entry ~ day * all_in_ramadan + (1 + day | p_id), data = model_df, family = binomial(link = "logit"), control = control)
plot_and_save(ramadan_model, "all_in_ramadan", "mlm_pipeline/outputs/ramadan_fit.png", list(`0` = "Not Ramadan", `1` = "Ramadan"))
capture.output(summary(ramadan_model), file = "ramadan_model.txt")
