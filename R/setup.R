if (!requireNamespace("pacman", quietly = TRUE)) {
     install.packages("pacman")
}

if (!requireNamespace("cli", quietly = TRUE)) {
     install.packages("cli")
}

cli::cli_alert_danger("Some packages may ask for the permission to install from source. Please select 'No'.")
cli::cli_alert_info("Loading/installing packages.")
pacman::p_load(renv, data.table, tidymodels, cli)

cli::cli_alert_success("setup.R finished.")
