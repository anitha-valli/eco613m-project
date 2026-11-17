### My Mini-Project



Hello,



This project is a replication of the paper by \[Damkliang \& Chumnaul (2025)][1](https://link.springer.com/article/10.1038/s41598-025-27294-1) titled \[Integrating quantile regression with ARIMA and ANN for interpretable and accurate PM2.5 forecasting in Hat Yai, Thailand] using Python.
The paper originally uses R. I do not claim any ownership of the original work. All credit for the original idea, methodology, and contributions goes to the original authors.



The work in this repository was done as part of coursework for the course "ECO613M - ECONOMIC DATA ANALYSIS" at Indian Institute of Technology (IIT), Kanpur (2025-2026 Semester 2) by Dr. Thirumulanathan Dhayaparan.

It is intended for educational purposes only.





This code can be directly downloaded and run from the terminal.



#### The Technical bits -



##### Pre-requisites:



1. Python installed (Python 3.13.1).
2. Install openpyxl (3.1.5), statsmodels, rpy2, scikit-learn and seaborn.
3. R installed along with some necessary packages mentioned below (R version 4.5.3 (2026-03-11 ucrt) -- "Reassured Reassurer") using the command install.packages("name\_of\_pkg")-

   * forecast
   * quantreg
4. Windows may require x64, x86 Visual C++ (ensure it is updated).
5. Modify the line r\_home = r"C:\\Program Files\\R\\R-4.5.3" in both the files as required (file path and R version).





The file titled \[Model.py] contains the modelling of all five models and the file titled \[Evaluation.py] contains the models applied to:



* a different time period at the same station (44T)
* three different stations (35T, 05T, 63T) during the same period



The above was also part of the paper [1](https://link.springer.com/article/10.1038/s41598-025-27294-1).





Under the Quantile Regression section, the 25th and 75th percentiles were also added for comparison.





When the files \[Model.py] and \[Evaluation.py] are run, they generate \[Full\_Data.xlsx] and \[Evaluation\_File.xlsx], respectively. The appropriate line may be commented to avoid this - these files are not strictly necessary for the code to run smoothly.





\[1]: https://link.springer.com/article/10.1038/s41598-025-27294-1



\-OR-



https://doi.org/10.1038/s41598-025-27294-1



DOI: 10.1038/s41598-025-27294-1

(08 December 2025, Volume 15, article number 43352)



**Thanks to everyone who supported this project.**

If you got till here, thank you for reading.

