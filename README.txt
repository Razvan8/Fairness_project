The requirments are in Requirments.txt. 

All the files that have to be run are in the folder notebooks. The files have to be run in the following order:

Data_prep_ML_German_dataset.ipynb
Fairness_analysis_German_dataset.ipynb
Data_prep_ML_German_biased.ipynb
Fairness_analysis_German_biased.ipynb


Data_prep_ML_Appendix_dataset.ipynb (I will not use the Appendix_dataset for my analysis, but I wanted to show that I tried that too and it was already fair)
Fairness_Analysis_Appendix_dataset.ipynb (I will not use the Appendix_dataset for my analysis, but I wanted to show that I tried that too and it was already fair)


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The folders contain the following:


Dataframes: it will store folders with dataframes used for fairness analysis and bias injection (These dataframes will be created and stored by running Data_prep_ML_German_biased.ipynb and Data_prep_ML_German_dataset.ipynb)

libs: it will contain the libraries that I created for this project (py files) and all the imports. Nothing has to be run in this folder.

ML_models: Here will be stored in specified folders the best ML model for different datasets for different levels of bias injection (These models will be created and stored by running Data_prep_ML_German_biased.ipynb and Data_prep_ML_German_dataset.ipynb) 

notebooks: Here will be the notebooks that create the results. There are 2 main types of notebooks. "Data_prep_ML" notebooks that prepare the data and store the suitable dataframes and ML models, and "Fairness analysis" notebooks that show the results for fairness analysis.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Remark: If one ones to be able to run directly the Fairness_analysis files without running the preprocessing and storing part (i.e Data_prep_ML files), please use this link https://github.com/Razvan8/Fairness_project that has the processed data already stored. It was too big to be attached on email. 
Remark: Even if one runs all files, it should not take more than 1 minute to run any of them.
Remark: By running the files in the specified order everything should run properly and the results should be the same at every run.  
Remark: The python version used is 3.11.3
