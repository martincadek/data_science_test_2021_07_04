# load data ---------------------------------------------------------------
# load cleaned data
(data_model <- fread("data/cleaned_USA_cars_datasets.csv"))
setDF(data_model)

cli_alert_info("Starting the model.R script, please wait...")

# split -------------------------------------------------------------------
# set seed to make the analysis reproducible
set.seed(062021)
data_split <- initial_split(data_model, strata = "price", prop = 0.75)
data_train <- training(data_split)
data_test  <- testing(data_split)

# cross-validation --------------------------------------------------------
# set how many folds
# no strata or repeats are determined
# basic cross-validation
cv_folds <- vfold_cv(data_train, v = 5)

# set prediction formula --------------------------------------------------
# define predictors or exploratory variables (all but price)
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
        # transform all categorical variables to dummy variables (one-hot encode)
        step_dummy(all_nominal()) %>% 
        # center numerical variables to a mean of zero 
        step_center(all_predictors()) %>%
        # standardize numerical variables to a standard deviation of one
        step_scale(all_predictors())



# define models -----------------------------------------------------------
# define random forest, mtry, trees, min_n are parameters that will vary in range
model_rf <- rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>% 
        set_mode("regression") %>%
        set_engine("ranger")

# define xgboost, mtry, trees, min_n are parameters that will vary in range
model_xgboost <- boost_tree(mtry = tune(), trees = tune(), min_n = tune()) %>% 
        set_mode("regression") %>%
        set_engine("xgboost")


# define pure ridge regression, this is done by setting mixture to 0
model_glmnet_ridge <- linear_reg(penalty = tune(), mixture = 0) %>% 
        set_mode("regression") %>%
        set_engine("glmnet")

# define pure lasso regression, this is done by setting mixture to 1
model_glmnet_lasso <- linear_reg(penalty = tune(), mixture = 1) %>% 
        set_mode("regression") %>%
        set_engine("glmnet")

# define simple lm model to compare with the other models
model_lm <- linear_reg() %>% 
        set_mode("regression") %>%
        set_engine("lm")

# define grids ------------------------------------------------------------
# define the ranges for all models that will be used during the tuning process
grid_rf <- grid_max_entropy(
        mtry(range = c(1, 20)),
        trees(range = c(500, 1000)),
        min_n(range = c(2, 10)),
        size = 30)

grid_xgboost <- grid_max_entropy(
        mtry(range = c(1, 20)),
        trees(range = c(500, 1000)),
        min_n(range = c(2, 10)),
        size = 60)

grid_glmnet_ridge <- grid_regular(
        penalty(range = c(-5, 5)), levels = 50)

# Set lower for lasso to avoid
# A correlation computation is required, but `estimate` is constant and has 0 standard
grid_glmnet_lasso <- grid_regular(
        penalty(range = c(-3, 3)), levels = 50)

# lm model is not tuned

# define workflows --------------------------------------------------------
# define workflows for each model which determines pre-processing and when 
# model is run

flow_rf <- workflow() %>% 
                add_recipe(default_preprocessing) %>%
                add_model(model_rf)
        
flow_xgboost <- workflow() %>% 
        add_recipe(default_preprocessing) %>%
        add_model(model_xgboost)

flow_glmnet_ridge <- workflow() %>% 
        add_recipe(default_preprocessing) %>%
        add_model(model_glmnet_ridge)

flow_glmnet_lasso <- workflow() %>% 
        add_recipe(default_preprocessing) %>%
        add_model(model_glmnet_lasso)

flow_glmnet_lm <- workflow() %>% 
        add_recipe(default_preprocessing) %>%
        add_model(model_lm)

# define metrics ----------------------------------------------------------
# set Root Mean Squared Error, R Squared, and Mean absolute error
# Note that ccc is a metric of both consistency/correlation and accuracy, 
# while metrics such as rmse are strictly for accuracy
report_metrics <- metric_set(rmse, rsq, mae)

# fit [train] -------------------------------------------------------------
# This is where all models are fitted and results from the fitting stored
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
        grid = grid_xgboost,
        metrics = report_metrics,
        control = control_grid(verbose = TRUE)
)

fit_glmnet_ridge <- tune_grid(
        flow_glmnet_ridge,
        resamples = cv_folds,
        grid = grid_glmnet_ridge,
        metrics = report_metrics,
        control = control_grid(verbose = TRUE)
)

fit_glmnet_lasso <- tune_grid(
        flow_glmnet_lasso,
        resamples = cv_folds,
        grid = grid_glmnet_lasso,
        metrics = report_metrics,
        control = control_grid(verbose = TRUE)
)

# LM is not using any tuning
fit_lm <- fit_resamples(
        flow_glmnet_lm,
        resamples = cv_folds,
        metrics = report_metrics,
        control = control_grid(verbose = TRUE)
)


# inspect and select ------------------------------------------------------
# This is useful only if inspected in interactive session

# collect_metrics(fit_rf)
# autoplot(fit_rf, metric = c("rmse", "rsq")) # won't work if the model was not tuned
# show_best(fit_rf, metric = "rmse")
# select_best(fit_rf, metric = "rmse")

# fit[test] ---------------------------------------------------------------
# select the best performing models and predict them on the test data
# RMSE is determined as the value according which to select the models
tuned_model_rf <- flow_rf %>% 
        finalize_workflow(select_best(fit_rf, metric = "rmse")) %>% 
        fit(data = data_train)

tuned_model_xgboost <- flow_xgboost %>% 
        finalize_workflow(select_best(fit_xgboost, metric = "rmse")) %>% 
        fit(data = data_train)

tuned_model_ridge <- flow_glmnet_ridge %>% 
        finalize_workflow(select_best(fit_glmnet_ridge, metric = "rmse")) %>% 
        fit(data = data_train)

tuned_model_lasso <- flow_glmnet_lasso %>% 
        finalize_workflow(select_best(fit_glmnet_lasso, metric = "rmse")) %>% 
        fit(data = data_train)

tuned_model_lm <- flow_glmnet_lm %>% 
        finalize_workflow(select_best(fit_lm, metric = "rmse")) %>% 
        fit(data = data_train)


# visualise ---------------------------------------------------------------
# visualise plots focusing on comparison of how models fit test data
# save these for later use in the report

# save the best models
df_best_models <-  select(data_test, price) %>%
        # select columns, take a model and fit it on test data,
        # rename predictions, bind all columns and them transform to long format
        bind_cols(
                predict(tuned_model_rf, new_data = data_test[, all_predictors]) %>%
                        rename(`Random Forest` = .pred),
                predict(tuned_model_xgboost, new_data = data_test[, all_predictors]) %>%
                        rename(`XGboost` = .pred),
                predict(tuned_model_ridge, new_data = data_test[, all_predictors]) %>%
                        rename(`Ridge Regression` = .pred),
                predict(tuned_model_lasso, new_data = data_test[, all_predictors]) %>%
                        rename(`Lasso Regression` = .pred),
                predict(tuned_model_lm, new_data = data_test[, all_predictors]) %>%
                        rename(`Linear Regression (OLS)` = .pred)
                ) %>%
        pivot_longer(!price, names_to = "model", values_to = "prediction")

# visualise the best models
gg_best_models <- df_best_models %>% 
                ggplot(aes(x = prediction, y = price)) + 
                geom_point(alpha = .4) + 
                geom_abline(col = "red") +
                scale_y_continuous(labels = scales::dollar_format()) +
                scale_x_continuous(labels = scales::dollar_format()) +
                facet_wrap(~model) + 
                labs(title = "Best models comparisons (Test data)", y = "Predicted price of used cars", x = "True price of used cars") +
                theme_minimal()

# save plot
ggsave(filename = "plots/gg_best_models.png", plot = gg_best_models, 
       width = 10, height = 5, dpi = 300, units = "in")

# provide table of metrics
df_best_models_metrics <- df_best_models %>%
        group_by(model) %>%
        metrics(price, prediction) %>%
        arrange(.estimate)

# additional visualisations -----------------------------------------------
# visualise the outcome variable
gg_explore_train <- data_train %>%
        ggplot(aes(price)) +
        geom_histogram(position = "identity", alpha = 0.5, bins = 20) +
        scale_x_continuous(labels = scales::dollar_format()) +
        labs(fill = NULL, x = "Price per car") +
        theme_minimal()

# save
ggsave(filename = "plots/gg_explore_train.png", plot = gg_explore_train, 
       width = 5, height = 5, dpi = 300, units = "in")

# save the random forest model (which performed the best)
the_best_model <- augment(tuned_model_rf, data_test)

# take metrics (for table)
the_best_model_metrics <- metrics(the_best_model, price, .pred)

# plot and save the best model (random forest)
gg_the_best_model <- the_best_model %>%
        ggplot(aes(price, .pred)) +
        geom_point(alpha = 0.2) +
        geom_abline(slope = 1, lty = 2, color = "red") +
        scale_y_continuous(labels = scales::dollar_format()) +
        scale_x_continuous(labels = scales::dollar_format()) +
        labs(title = "Best model predictions (Random Forest)", 
             x = "True price of used cars", 
             y = "Predicted price of used cars") +
        theme_minimal()

# save the model
ggsave(filename = "plots/gg_the_best_model.png", plot = gg_the_best_model, 
       width = 5, height = 5, dpi = 300, units = "in")


cli_alert_success("The model.R script, has finished running.")
