# load data ---------------------------------------------------------------
(data_model <- fread("data/cleaned_USA_cars_datasets.csv"))
setDF(data_model)

# split -------------------------------------------------------------------
# set seed 
set.seed(062021)
data_split <- initial_split(data_model, strata = "price", prop = 0.7)

data_train <- training(data_split)
data_test  <- testing(data_split)

# set predictors ----------------------------------------------------------
all_predictors <- c("mileage", "year", "title_status", "state", 
                "brand_model", "condition_minutes_left", "color_grouped")

full_formula <- as.formula(price ~ mileage + year + title_status + state + 
                                brand_model + condition_minutes_left + color_grouped)


# model - ranger ----------------------------------------------------------
default_mtry <- floor(sqrt(length(all_predictors)))

rf_settings <- rand_forest(mode = "regression", mtry = default_mtry, trees = 1000)

rf_model <- rf_settings %>%
     set_engine("ranger") %>%
     fit_xy(
          x = data_train[, all_predictors],
          y = data_train$price
          )

rf_model

# evaluation
test_fits <- data_test %>%
     select(price) %>%
     bind_cols(
          predict(rf_model, new_data = data_test[, all_predictors])
     )

test_fits %>% metrics(truth = price, estimate = .pred) 

# model - glmnet ----------------------------------------------------------
glmnet_recipe <- recipe(
     full_formula, 
     data = data_train) %>%
     step_other(brand_model) %>%
     step_dummy(all_nominal()) %>%
     step_center(all_predictors()) %>%
     step_scale(all_predictors()) %>%
     prep(training = data_train, retain = TRUE)

glmn_model <- linear_reg(penalty = 0.001, mixture = 0.5) %>% 
     set_engine("glmnet") %>%
     fit(price ~ ., data = bake(glmnet_recipe, new_data = NULL))

glmn_model


# combine models ----------------------------------------------------------
test_glmnet <- bake(glmnet_recipe, new_data = data_test, all_predictors())

test_fits <- test_fits %>%
     rename(`random forest` = .pred) %>%
     bind_cols(
          predict(glmn_model, new_data = test_glmnet) %>%
               rename(glmnet = .pred))

test_fits %>% metrics(truth = price, estimate = .pred) 



# compare models ----------------------------------------------------------
test_fits %>% 
     pivot_longer(!price, names_to = "model", values_to = "prediction") %>%
     ggplot(aes(x = prediction, y = price)) + 
     geom_abline(col = "green", lty = 2) + 
     geom_point(alpha = .4) + 
     facet_wrap(~model) + 
     coord_fixed()

