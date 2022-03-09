# Kaggle Competition: **Housing Prices Competition** (Ridge)

## **Summary**

### **Introduction**

This is a predictive analysis for housing price analysis, one of the ongoing community prediction competitions on Kaggle. ([https://www.kaggle.com/c/home-data-for-ml-course/overview](https://www.kaggle.com/c/home-data-for-ml-course/overview))

I used and practiced Ridge Regression for this competition as it contains a lot of x variables having high correlation to target variable(housing price).

![**Correlation between major X variables and y variable**](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/7ccee6fe-5598-41ce-bc48-e58bc206d321/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220309%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220309T185223Z&X-Amz-Expires=86400&X-Amz-Signature=4a0fb1fc6f745b303bb84b8b1773108b7bf55cdfd85d839a2ad139be8a28fcf2&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

**Correlation between major X variables and y variable**

### **Datasets**

Train Data → 79 X variables related to houses and sales prices (y variable). Total 1,460 rows.

Test Data → 79 X variables related to houses. Total 1,459 rows.

### **Language and libraries**

Python, Pandas, Sklearn, Seaborn

### **Data Preprocessing**

**Outliers** 

I found some ‘bad’ outliers that deviated from overall trend in ‘GrLivArea’ and ‘GarageArea’ columns which are major continuous variable in the datasets. As I am going to use Linear Regression model with Ridge, I considered to delete these outliers because they can provide wrong information about the columns and make receive more penalty when the model is fitting. Test score was improved when these outliers are removed.

![                Bad outliers in ‘GrLivArea’ column](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/839bc1d6-52ac-4171-8f4d-32bf83e6f2fa/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220309%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220309T185238Z&X-Amz-Expires=86400&X-Amz-Signature=af430fcac12679073672cd47c982aade001a1bce1daa80df5b976623e9575f13&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

                Bad outliers in ‘GrLivArea’ column

![              Bad outliers in ‘GarageArea’ column](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/c8e219d1-eda9-4c0c-bec8-f4fb6849c714/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220309%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220309T185250Z&X-Amz-Expires=86400&X-Amz-Signature=82d4bcb9686fa71e6195adfd2c735ab84f71edabe5cef7bbf03c4cb836a1cad0&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

              Bad outliers in ‘GarageArea’ column

**Imputation** 

Some columns(’Alley’, ‘PoolQC’, ‘Fence‘ and ‘MiscFeature’) having more than 60% of missing values are removed as there are too small number of data which can be used as reference for imputation. I used ‘mean’ value for missing values in numerical columns. Also, missing values in categorical columns are filled with ‘-1’ to mark that missing values are different from current categories.

**One-Hot Encoding** 

I used ‘get_dummies’ method in Pandas for One-Hot-Encoding for categorical columns. One-Hot-Encoding is also applied to ‘MSSubClass’ column which has numerical categories.

**Scaling** 

It is important to scale so that the penalty is given fairly for each column when we are using Ridge Regression. Therefore, I used Standard Scaler for scaling.

**Taking log for y variable**

y variable’s distribution is right-skewed. Therefore, I took log for y values to make it has more balanced distribution. I tested both versions(using raw value and value after taking log) of model and discovered taking log was helpful to improve accuracy of the model.

![y variable’s distribution before taking log](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/715077d8-04b4-4d48-a9ac-38e6b81fcbe2/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220309%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220309T185304Z&X-Amz-Expires=86400&X-Amz-Signature=368b8a3805df559138fdd13c54fcfc571b6e749dc807236c764a52a58c3fd3c9&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

y variable’s distribution before taking log

![The distribution after taking log](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/790c6fc1-f49f-41d9-8368-69480cf9b30f/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220309%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220309T185314Z&X-Amz-Expires=86400&X-Amz-Signature=cdeb54ce839ec1bf0da9ec5588addb3a4d7a4b1f423dbad78bccdd55c8f07553&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

The distribution after taking log

### **Model Selection**

**Model** 

As explained above, Ridge Regression is selected for this dataset because there are a lot of columns having high correlation with y variable. However, initial modelling with CatBoost also showed remarkable performance(rank: 191/34213), but I decided to use Ridge to learn and practice more with Linear Regression model and required techniques for this model.

**Hyperparameter Tuning**

I tuned ‘Alpha’, the most important parameter for Ridge model, to optimize the model performance by using range from 10 to 1000 and cross validation. As the train dataset’s size is not big enough(1,460 rows) I used 10 folds for cross validation. 

200 for Alpha showed lowest mean RMSE for validation datasets, therefore, this value is used for the final model.

![Result for Hyperparameter tuning (Key: value for Alpha, Value: mean RMSE for validation datasets)](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/0dad46cc-582b-4ba5-9a0f-afbe25d5e6fb/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220309%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220309T185328Z&X-Amz-Expires=86400&X-Amz-Signature=3f070366aa5d1c84ae1d2d1b5681593a660db32f40e70cc01a62c80f8cb9d418&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

Result for Hyperparameter tuning (Key: value for Alpha, Value: mean RMSE for validation datasets)

### **Evaluation**

The competition’s evaluation metric is RMSE. 

Final model’s Train RMSE was 16662 and Public score(RMES) for test dataset after submission was 13923. (rank: 353/34213)

### **Takeaway**

**Linear Regression can be powerful:**

As a beginner, I was highly depending on decision tree based models such as Random Forest, Cat Boost, XGBoost. However, I learned that linear regression model can be also very effective if I can use necessary techniques including scaling, pre-processing for outliers and proper one-hot-encoding. Also, this project became an opportunity to review concept of ‘Ridge’ model and how it works.
