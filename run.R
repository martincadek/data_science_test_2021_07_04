sapply(list("R/setup.R", "R/clean.R"), source)
sapply(list("R/model.R"), source)
rmarkdown::render(input = "R/report.Rmd")