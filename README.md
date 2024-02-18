# Real-Estate-Price-Prediction
Real Estate Price Prediction Kaggle Competition

1. Real Estate price prediction competition comprises of predict prices of houses based on a combination of various parameters using Regression techniques.

2. The Dataset comprised of Training and Testing data. Testing data does not include Sale price of houses, but, it includes all variables in the Training set.

3. Training Data set comprises of numerical, categorical and ordinal data.

4. Numerical Data includes variables such as (Only key variables are listed her):
  a. Overall plot area
  b. Living area
  c. Basement area
  d. Garage cars, etc.
  
5. Categorical variables include:
  a. Location
  b. Pool QC
  c. House Style
  d. Sale type,etc.
  
 6. Ordinal variables include:
  a. House Overall quality
  b. House External condition,etc.
  
 7. Model uploaded here is a WIP version comprising of 14 sections. Key features of model are descriptive and visual analysis, feature engineering and XGB Regressor.Each individual
section comprises of changes on training data followed by replication on testing data wherever applicable. 
 
 8. Section 1 : Comprises of importing packages for the model
 
 9. Section 2: Import data and basic checks on imported data
 
 10. Section 3: Delete columns based on low count of available data as per section 2
 
 11. Section 4: Comprises of plotting sales price trend wrt categorical variables. This is useful for feature engineering
 
 12. Section 5: Comprises of plotting sales price trend wrt numberical data.
 
 13. Section 6: Outlier rows are deleted based on visualization of section 5 and analysis in section 5.
 
 14. Section 7: Comprises of deleting columns based on NA cells and replacing NA with required value as per variable type.
 
 15. Section 8: Categorical columns for Year values are converted to a numerical feature of age. Followed by this columns are deleted to save on memory.
 
 16. Section 9: Dummy generation and scaling has been carried out in this section
 
 17. Section 10: Splitting of training data set into test and train for modeling purpose alongwith shuffling of data is carried out in this section
 
 18. Section 11:Data preparation for XGB Regressor model is carried out here.
 
 19. Section 12: XGB Regressor model has been implemented in this section
 
 20. Section 13: Model accuracy analysis and prediction on Testing data for submission
 
 21. Section 14: Conversion to CSV file for submission is carried out in this section
 
 22. Model Remarks: 
 a. Model has been built to prepare an overall framework for XGB Regressor based prediction 
 b. Model has further scope for improvement in feature analysis, paramter and hyper-parameter tuning. 
 c. Moreover, current version does not support feature name extraction for further analysis and refinement. 
 d. Also, use of K-Fold method will also be helpful in improving training of model
 
 
 
  


  
