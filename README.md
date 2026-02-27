# Assignment-4

Purpose
This project analyzes NBA historical data to identify trends in player longevity and shooting efficiency. By calculating the "average accuracy" through the integration of a regression line rather than a simple mean, we can account for the overall trajectory of a player's career.

Implementation Detail
Manual OLS: Regression is performed using the covariance-variance method.
Average Value Theorem: Used to determine expected accuracy by integrating the fit line over the player's active span.
Statistical Battery: Implementation of 3rd and 4th standardized moments to describe the distribution of Field Goals.

Limitations
Data Density: Players with very few 3PT attempts may have volatile "Accuracy" percentages that skew the linear fit.
Relational Dependency: The T-test assumes normal distribution; however, NBA shot data is typically right-skewed.
