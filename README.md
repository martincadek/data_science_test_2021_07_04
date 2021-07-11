# Run
To run the code in the repository, please use RStudio and open the project using data_science_test_2021_07_04.Rproj file.

After opening the project in RStudio, please navigate to the master file (located in the parent folder or first level of this repository) called run.R and source or run all of the lines.

The code initiates directory setup, clean data, and run several algorithms. The algorithms may take some time to run depending on your CPU as cross-validation, and varying tuning parameters are utilised.

Once the code has finished running, you can find the final report located in R/report.html.
The script also outputs several graphical outputs into plots/.

Packages are version controlled using renv, which should be triggered automatically if the project is opened. Rproj file. However, in some cases, the users may be prompted to install dependencies.

# Objective
From the supplied dataset (USA_cars_datasets.csv) build a model to predict the price of a used car given the other information available in the data. The final model should be evaluated to determine the expected level of predictive performance.