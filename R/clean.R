# load data ---------------------------------------------------------------
cli_alert_info("Loading data...")
(data_usa_cars <- fread("data/USA_cars_datasets.csv"))

# What's what:
# price = Can or USD dollars
# brand = car
# model = car
# title_status = clean / pay for damages
# mileage
# color = 49 types
# vin = vehicle identification number
# lot = manufacturer number
# state = USA / Canada states
# country = USA/CAN
# condition = auction time

# explore / wrangle -------------------------------------------------------
cli_alert_info("Exploring data...")
columns_to_drop <- c("V1") # remove row col
data_usa_cars[, (columns_to_drop) := NULL]

# save default numeric and characters
columns_numeric <- names(data_usa_cars)[sapply(data_usa_cars, is.numeric)]
columns_character <- names(data_usa_cars)[sapply(data_usa_cars, is.character)]

# Assessing features for missing and unique values
sapply(data_usa_cars, function(x){sum(is.na(x))}) # NAs
sapply(data_usa_cars, function(x){uniqueN(x)}) # duplicated
data_usa_cars[which(duplicated(vin)),] # duplicated extract

# Looking for unusual values
data_usa_cars[price < 100, .N]
data_usa_cars[mileage < 10,.N]
data_usa_cars[year < 1990,.N]
sapply(data_usa_cars[, ..columns_character], table)
sapply(data_usa_cars[, ..columns_numeric], function(x){c(mean = mean(x), sd = sd(x), min = min(x), max = max(x))})
cube(data_usa_cars, j = list(count = .N), by = c("color"), id = TRUE)[order(-count)] # investigating colours
summary(data_usa_cars)

# clean -------------------------------------------------------------------
cli_alert_info("Cleaning data...")
# remove duplicates
cli_alert("Removing duplicate observations...")
data_usa_cars <- unique(data_usa_cars, by = c("vin", "lot"))

cli_alert("Removing outlier observations...")
# remove canada
data_usa_cars <- data_usa_cars[country != "canada",]

cli_alert("Preparing new columns: brand_model, condition_expired, condition_minutes_left, color_grouped")
# prepare data
data_usa_cars[, `:=`(
     # merge brand + model
     brand_model = paste(brand, model, sep = "_"),
     # transform minutes
     # Line 70 replaces "Condition Expired" as 0
     condition_expired = ifelse(condition == "Listing Expired", TRUE, FALSE),
     condition_minutes_left = time_length(duration(gsub(x = condition, pattern = " left", "")), unit = "minute"),
     # group colours to those that are common and "other" (factorise)
     # this is optional and I'll see if this is better than letting the algorithm decide how to group colours
     color_grouped = fcase(
          color == "white", "white",
          color == "black", "black",
          color == "gray", "gray",
          color == "silver","silver",
          color == "no_color", "no_color",
          default = "other_colors"
     ),
     year = as.factor(year)
     )]
# set condition that expired from NA to 0 (NAs due to coercion)
data_usa_cars[, condition_minutes_left := ifelse(is.na(condition_minutes_left), 0, condition_minutes_left)]

# move those valued zero aside (or similar), also those that are historical, and probably those that are salvage insurance
potential_outliers <- data_usa_cars[price <= 10 | title_status == "salvage insurance" | mileage < 2]
# remove those that do not have any value or mileage
cli_alert("Removing observations with null price and mileage...")
data_usa_cars <- data_usa_cars[price > 0][mileage > 0]

# save --------------------------------------------------------------------
fwrite(data_usa_cars, file = "data/cleaned_USA_cars_datasets.csv")
fwrite(potential_outliers, file = "data/potential_outliers.csv")

cli_alert_info("Cleaning workspace...")
rm(columns_character, columns_numeric, columns_to_drop, data_usa_cars, potential_outliers)

cli_alert_success("clean.R finished")