# load data ---------------------------------------------------------------
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
columns_to_drop <- c("V1") # remove row col
data_usa_cars[, (columns_to_drop) := NULL]
columns_numeric <- names(data_usa_cars)[sapply(data_usa_cars, is.numeric)]
columns_character <- names(data_usa_cars)[sapply(data_usa_cars, is.character)]


lapply(data_usa_cars, function(x){sum(is.na(x))})
lapply(data_usa_cars, function(x){length(unique(x))})
lapply(data_usa_cars, function(x){sum(duplicated(x))})
data_usa_cars[which(duplicated(vin)),]
data_usa_cars[which(price < 100),]
data_usa_cars[which(mileage < 10),]
lapply(data_usa_cars[, ..columns_character], table)
summary(data_usa_cars)




