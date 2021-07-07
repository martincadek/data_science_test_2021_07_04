# WIP - implement parallel processing

# load data ---------------------------------------------------------------
(data_model <- fread("data/cleaned_USA_cars_datasets.csv"))
setDF(data_model)

# split -------------------------------------------------------------------
# set seed 
set.seed(062021)
data_split <- initial_split(data_model, strata = "price", prop = 0.75)
data_train <- training(data_split)
data_test  <- testing(data_split)

# cross-validation --------------------------------------------------------
cv_folds <- vfold_cv(data_train, v = 5)

# set prediction formula --------------------------------------------------
all_predictors <- c("mileage", "year", "title_status", "state", 
                    "brand_model", "condition_minutes_left", "color")

full_formula <- as.formula(price ~ mileage + year + title_status + state + 
                                   brand_model + condition_minutes_left + color)

# pre-processing ----------------------------------------------------------
default_preprocessing <- recipe(
        full_formula, 
        data = data_train) %>%
        # combine factors with low frequency to other
        step_other(year) %>%
        step_other(state) %>% 
        step_other(brand_model) %>% 
        step_other(color) %>% 
        # makes all categorical variables a dummy variables
        step_dummy(all_nominal()) %>% 
        # centering to a mean of zero 
        step_center(all_predictors()) %>%
        # standardizing to a standard deviation of one
        step_scale(all_predictors())



# define models -----------------------------------------------------------
model_rf <- rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>% 
        set_mode("regression") %>%
        set_engine("ranger")

model_xgboost <- boost_tree(mtry = tune(), trees = tune(), min_n = tune()) %>% 
        set_mode("regression") %>%
        set_engine("xgboost")

# ridge regularization = 0
# lasso regularization = 1
model_glmnet <- linear_reg(penalty = tune(), mixture = 0) %>% 
        set_mode("regression") %>%
        set_engine("glmnet")

# define ols, define pls

# define grids ------------------------------------------------------------
grid_rf <- grid_max_entropy(
        mtry(range = c(1, 20)),
        trees(range = c(500, 1000)),
        min_n(range = c(2, 10)),
        size = 30)

grid_glmnet <- grid_regular(
        penalty(range = c(-5, 5)), levels = 50)

grid_xgboost <- grid_max_entropy(
        mtry(range = c(1, 20)),
        trees(range = c(500, 1000)),
        min_n(range = c(2, 10)),
        size = 60)

# define workflows --------------------------------------------------------
flow_rf <- workflow() %>% 
                add_recipe(default_preprocessing) %>%
                add_model(model_rf)
        
flow_xgboost <- workflow() %>% 
        add_recipe(default_preprocessing) %>%
        add_model(model_xgboost)

flow_glmnet <- workflow() %>% 
        add_recipe(default_preprocessing) %>%
        add_model(model_xgboost)



# define metrics ----------------------------------------------------------
report_metrics <- metric_set(rmse, rsq, ccc)

# fit [train] -------------------------------------------------------------
fit_rf <- tune_grid(
        flow_rf,
        resamples = cv_folds,
        grid = grid_rf,
        metrics = report_metrics,
        control = control_grid(verbose = TRUE)
)

fit_xgboost <- tune_grid(
        flow_xgboost,
        resamples = cv_folds,
        grid = grid_rf,
        metrics = report_metrics,
        control = control_grid(verbose = TRUE)
)

fit_glmnet <- tune_grid(
        flow_glmnet,
        resamples = cv_folds,
        grid = grid_rf,
        metrics = report_metrics,
        control = control_grid(verbose = TRUE)
)

# inspect and select ------------------------------------------------------
fit_rf
collect_metrics(fit_rf)
autoplot(fit_rf, metric = "rmse")
show_best(fit_rf, metric = "rmse")
select_best(fit_rf, metric = "rmse")

fit_xgboost
collect_metrics(fit_xgboost)
autoplot(fit_xgboost, metric = "rmse")
show_best(fit_xgboost, metric = "rmse")
select_best(fit_xgboost, metric = "rmse")

fit_glmnet
collect_metrics(fit_glmnet)
autoplot(fit_glmnet, metric = "rmse")
show_best(fit_glmnet, metric = "rmse")
select_best(fit_glmnet, metric = "rmse")


# fit[test] ---------------------------------------------------------------
tuned_model_rf <- flow_rf %>% 
        finalize_workflow(select_best(rf_fit, metric = "rmse")) %>% 
        fit(data = data_train)

predict(tuned_model_rf, data_test)

augment(tuned_model_rf, new_data = data_test) %>%
        rmse(truth = price, estimate = .pred)


# visualise ---------------------------------------------------------------
df_best_models <-  select(data_test, price) %>%
        bind_cols(
                predict(tuned_model_rf, new_data = data_test[, all_predictors]) %>%
                        rename(`Random Forest` = .pred)
                ) %>%
        pivot_longer(!price, names_to = "model", values_to = "prediction")

(gg_best_models <- df_best_models %>% 
        ggplot(aes(x = prediction, y = price)) + 
        geom_abline(col = "red") + 
        geom_point(alpha = .4) + 
        facet_wrap(~model) + 
        coord_fixed() +
        labs(title = "Best models comparisons", y = "Price", x = "Prediction") +
        theme_minimal())



# DEP BELOW ---------------------------------------------------------------
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
glmn_model <- linear_reg(penalty = 0.001, mixture = 0.5) %>% 
     set_engine("glmnet") %>%
     fit(price ~ ., data = bake(default_preprocessing, new_data = NULL))

glmn_model



# model - xgboost ---------------------------------------------------------
# Specify model
xgboost_model <- 
        boost_tree(
                mode = "regression",
                trees = 1000,
                min_n = tune(),
                tree_depth = tune(),
                learn_rate = tune(),
                loss_reduction = tune()
        ) %>%
        set_engine("xgboost", objective = "reg:squarederror") %>%
        fit(price ~ ., data = bake(default_preprocessing, new_data = NULL))

# Specify grid
xgboost_params <- 
        parameters(
                min_n(),
                tree_depth(),
                learn_rate(),
                loss_reduction()
        )

xgboost_grid <- 
        dials::grid_max_entropy(
                xgboost_params, 
                size = 60
        )

xgboost_grid

# combine models ----------------------------------------------------------
test_glmnet <- bake(default_preprocessing, new_data = data_test, all_predictors())

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