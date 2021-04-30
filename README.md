# Data Science_Project

***

##  Introduction
When people buy used cars, it is very important to buy cars in good condition at reasonable prices. However, most of the consumer damage relief measures related to used cars, which were released by the Korea Consumer Affairs Agency from 2016 to June 2019, were “in many cases, the state of the vehicle is different from what was previously announced”. 
So, when people buy a used car, they can use we created a program that predicts condition of the used car by entering information such as the mileage, year, manufacturer, etc.


***

## Data Description
### Data source
We used 'Used cars dataset' registered in Kaggle.
This data is scraped every few months, it contains most all relevant information that Craigslist provides on car sales including columns like price, condition, manufacturer, latitude/longitude, and 18 other categories.

### "Vehicles.csv" Original Data Description
size: 25 columns X 539759 row
columns = [“id”, “url”, “region”, “region_url”, “price”, “year”, “manufacturer”, “model”, “condition”, “cylinders”, “fuel”, “odometer”, “title_status”, “transmission”, “vin”, “drive”, “size”, “type”, “paint_color”, “image_url”, “description”, “country”, “state”, “lat”, “long”]

### “Vehicles (1).csv” Data Inspection
Select 6 columns to use for data analysis (including target value).
![image](https://user-images.githubusercontent.com/57340671/114291827-72314b00-9ac5-11eb-8c92-0e9c7220016c.png)
![image](https://user-images.githubusercontent.com/57340671/114291829-73fb0e80-9ac5-11eb-8569-225339d0193a.png)

=>

![image](https://user-images.githubusercontent.com/57340671/114291831-765d6880-9ac5-11eb-9fb1-437eebdc24d0.png)

Used columns = [“price”, “year”, “manufacturer”, “condition”, “cylinders”]

Result of column extraction 

![image](https://user-images.githubusercontent.com/57340671/114291833-79585900-9ac5-11eb-9293-46a6953135f8.png)
![image](https://user-images.githubusercontent.com/57340671/114291849-99881800-9ac5-11eb-9195-521660805006.png)
![image](https://user-images.githubusercontent.com/57340671/114291850-9b51db80-9ac5-11eb-8b08-d75e1808557d.png)
![image](https://user-images.githubusercontent.com/57340671/114291852-9d1b9f00-9ac5-11eb-928f-6fde36409cf8.png)
![image](https://user-images.githubusercontent.com/57340671/114291856-a442ad00-9ac5-11eb-9e52-6928a6e25c93.png)
![image](https://user-images.githubusercontent.com/57340671/114291859-a6a50700-9ac5-11eb-94f9-fb2fba9abcc4.png)
![image](https://user-images.githubusercontent.com/57340671/114291861-a7d63400-9ac5-11eb-9be8-4a4d7d37d4af.png)


***

## Data Preprocessing
### Handle “Null” Condition
There were 44 percent missing value in the target feature ‘Condition’. If we start the analysis in this state, or use half the remaining data to fill 44% of the missing values, we thought there would be confusion in the subsequent data analysis.
Therefore, we proceeded with the deletion of rows containing 'missing value' in 'condition' with the aim of having accurate data and giving the user the appropriate predicted results.
<Open vehicles (1).csv & Drop rows with missing values in ‘Condition’>
![image](https://user-images.githubusercontent.com/57340671/114291892-df44e080-9ac5-11eb-9fde-4c53e64b3b49.png)
![image](https://user-images.githubusercontent.com/57340671/114291895-e10ea400-9ac5-11eb-8d2d-a2e168cd0907.png)

<'condition' data status after the above process>

![image](https://user-images.githubusercontent.com/57340671/114291896-e23fd100-9ac5-11eb-9bf6-cb66a1307f5a.png)

### Handle Dirty data
'Box plot' was used to view large amounts of data at a glance. We used this to check the outlier, and there were quite a few outliers. Therefore, it was determined that converting 'dirty data' to 'null' and using existing data to populating it than deleting it later would be more helpful for future predictions.
The following is the code and result is expressing the data of the 'price' column using a box plot

![image](https://user-images.githubusercontent.com/57340671/114291920-f552a100-9ac5-11eb-895f-126eacb4b823.png)
![image](https://user-images.githubusercontent.com/57340671/114291921-f7b4fb00-9ac5-11eb-84ea-22aa09c78bed.png)

It is the code and result that deleting the outlier exists in 'price' using 'IQR'.

![image](https://user-images.githubusercontent.com/57340671/114291923-f8e62800-9ac5-11eb-93a9-b5a5a7395c83.png)
![image](https://user-images.githubusercontent.com/57340671/114291925-fa175500-9ac5-11eb-9778-849ad48db957.png)

‘The following is the code and result is expressing the data of the ‘odometer’ column using a box plot

![image](https://user-images.githubusercontent.com/57340671/114291929-0a2f3480-9ac6-11eb-9557-1caf086c5794.png)
![image](https://user-images.githubusercontent.com/57340671/114291931-0b606180-9ac6-11eb-9fdc-0562f29909fb.png)

It is the code and result that deleting the outlier exists in 'odometer' using 'IQR'.

![image](https://user-images.githubusercontent.com/57340671/114291932-0c918e80-9ac6-11eb-8d46-eefbb6d0c551.png)
![image](https://user-images.githubusercontent.com/57340671/114291933-0dc2bb80-9ac6-11eb-98df-28e73dcd1922.png)

This is the result that has changed since the 'dirty data' processing was completed.

![image](https://user-images.githubusercontent.com/57340671/114291935-0ef3e880-9ac6-11eb-8db9-4639b34fe7a5.png)

=>

![image](https://user-images.githubusercontent.com/57340671/114291940-10bdac00-9ac6-11eb-87e7-24449524846c.png)

### Fill missing value
Missing value processing is essential for accurate target value prediction. After finding the missing value in the dataset, based on the target condition value, we divided it by condition and replaced the categorical feature with the mode value, and set limit=2 to drop the rest. The columns other than the categorical feature were replaced by the mean value for each condition.	
First, change the empty data to missing value and check how many missing values are in each column.

![image](https://user-images.githubusercontent.com/57340671/114291956-30ed6b00-9ac6-11eb-8916-37a9b2ff0957.png)
![image](https://user-images.githubusercontent.com/57340671/114291957-321e9800-9ac6-11eb-8522-28ad1bfb116c.png)

Next, change the cylinder column which has missing value from '8 cylinder' to '8 ' so that the data type of column can be calculated. Then, obtain the mean value for each condition and replace the missing value.

![image](https://user-images.githubusercontent.com/57340671/114291961-39de3c80-9ac6-11eb-9ce1-09623cdac43a.png)
![image](https://user-images.githubusercontent.com/57340671/114291963-3b0f6980-9ac6-11eb-9cdb-825c0385748d.png)
![image](https://user-images.githubusercontent.com/57340671/114291964-3c409680-9ac6-11eb-80fa-0228f244bebf.png)

Finally, obtain the mode value for each condition and replace the missing value at the manufacturer column which has missing value. Also, set limit=2 and drop the rest.

![image](https://user-images.githubusercontent.com/57340671/114291966-46fb2b80-9ac6-11eb-8a5c-b36ab6693dec.png)
![image](https://user-images.githubusercontent.com/57340671/114291967-482c5880-9ac6-11eb-8cfc-892330faa9aa.png)

### Normalization
After processing ‘Dirty data' and 'Missing value', we standardized using 'min-max normalization' to make 'features' scale equal. At this time, the value of 'categorical value' consisting of characters was quantified using 'Label Encoder'.
Then, we proceeded min-max normalization and completed data preprocessing by converting the data into a value between 0 and 1.

![image](https://user-images.githubusercontent.com/57340671/114291974-57aba180-9ac6-11eb-9db9-d91cbbc785e8.png)
![image](https://user-images.githubusercontent.com/57340671/114291975-58dcce80-9ac6-11eb-9e76-17a1c50a743c.png)

< Results using 'Label Encoder' >

![image](https://user-images.githubusercontent.com/57340671/114291976-5a0dfb80-9ac6-11eb-9ec5-60a8acdf5390.png)

<Result standardized data>

![image](https://user-images.githubusercontent.com/57340671/114291981-6abe7180-9ac6-11eb-90ed-029bf76a2c2e.png)


***

## Algorithms
### Multiple linear regression
We use multiple linear regression to predict the target value condition because the used car dataset have to predict with multiple columns. Therefore, after dividing the columns except condition into train and test, the results for multiple linear regression were printed as a table for the summary by comparing the predicted values with the condition column.
First, Represent the raw data as a matrix by correlation matrix.

![image](https://user-images.githubusercontent.com/57340671/114291991-80339b80-9ac6-11eb-83ce-7bd847d430b7.png)
![image](https://user-images.githubusercontent.com/57340671/114291992-8164c880-9ac6-11eb-9f11-afea435fbd17.png)
![image](https://user-images.githubusercontent.com/57340671/114291993-8295f580-9ac6-11eb-872a-9afe1c7f38f2.png)

Next, the condition column is placed in y and the rest of the column in x, and the two are divided into train and test, and then learned through linear model. 

![image](https://user-images.githubusercontent.com/57340671/114291999-92153e80-9ac6-11eb-95ff-aef6b90a8e17.png)

Create an OLS model and fit data to show the results of the multiple linear regression model in detail. And check for the normality of the residuals with the graph in below.

![image](https://user-images.githubusercontent.com/57340671/114292002-93df0200-9ac6-11eb-8264-00e4c9bd9042.png)
![image](https://user-images.githubusercontent.com/57340671/114292003-95102f00-9ac6-11eb-8904-20acf4394a08.png)

Finally, to evaluate the accuracy of the model, RMSE, mean absolute error and mean squared error is calculated, and the results of the OLS model are output in summary. 

![image](https://user-images.githubusercontent.com/57340671/114292012-a3f6e180-9ac6-11eb-9fa5-3960b346d7fa.png)
![image](https://user-images.githubusercontent.com/57340671/114292014-a5280e80-9ac6-11eb-9cb1-b7cb46d8d1e4.png)
![image](https://user-images.githubusercontent.com/57340671/114292016-a6593b80-9ac6-11eb-88b7-fd07001107e2.png)

The accuracy of the multiple linear regression was low because, as you can see from the graph above, the accuracy of the blue dots in the front part was low because they were far from the red line so it shows the dataset contain anomaly value. Also, because the target value, condition column is a categorical feature, the accuracy of the multiple linear regression was low.

### KNN - Classification
'used car dataset' is the data that the target already has a class as 'catalogical value'. So we decided to use classification, of which we used Knn-classification.
The accuracy of Knn classification depends on the k value. So, to find the optimal k value, we calculated the accuracy of the case where k is 1 to 15 and plotted it on a graph.

< Code for finding optimal k values >

![image](https://user-images.githubusercontent.com/57340671/114292030-bcff9280-9ac6-11eb-9016-554f889b7d86.png)

< optimal k accuracy result graph >

![image](https://user-images.githubusercontent.com/57340671/114292031-bec95600-9ac6-11eb-88c2-a6df1e05316b.png)

The graph shows that the smaller the k value, the higher the accuracy.
However, when k=1, it is said that the best case is very special. So We looked at why these results came out.

![image](https://user-images.githubusercontent.com/57340671/114292035-c1c44680-9ac6-11eb-8814-c8950e568d0a.png)

After looking at the various data, We saw an article saying, "If sufficient data exists, the accuracy of 1NN can be good. 
Based on this article, the first result is as follows. "Because of the large amount of data sets we used." In fact, if you look at the data, "excellent", "good", which accounts for most of the total condition values, there were many more than other condition values. So when you look for close neighbor values, the above two targets are more likely to be predicted and already more distributed, so the greater the k, the more confusion the greater the accuracy is, and the higher the k=1, the higher the accuracy.	
Our predictions may be wrong. But through this project, I think I could learn a lot by thinking about this.
Anyway, the result of setting k as 1 and applying KNN classification was 68% accuracy.

![image](https://user-images.githubusercontent.com/57340671/114292044-db658e00-9ac6-11eb-8036-1af6f183029d.png)
![image](https://user-images.githubusercontent.com/57340671/114292045-dc96bb00-9ac6-11eb-8952-7ab6c17d33fb.png)


***
## Evaluation
### K-Fold cross validation
- About Multiple linear regression

Since multiple linear regression is not appropriate when the target value is categorical, the prediction accuracy is low, so it was confirmed that the average accuracy of cross-validation was low when the 5th cross-validation was performed.

![image](https://user-images.githubusercontent.com/57340671/114292064-fcc67a00-9ac6-11eb-9535-0490b5b6562d.png)
![image](https://user-images.githubusercontent.com/57340671/114292068-ff28d400-9ac6-11eb-8a25-b43372374b43.png)
![image](https://user-images.githubusercontent.com/57340671/114292070-005a0100-9ac7-11eb-8e33-344a021ceb0c.png)
![image](https://user-images.githubusercontent.com/57340671/114292071-018b2e00-9ac7-11eb-9d1e-774c7867149a.png)
![image](https://user-images.githubusercontent.com/57340671/114292073-03ed8800-9ac7-11eb-9791-8e9e79229aba.png)
![image](https://user-images.githubusercontent.com/57340671/114292075-05b74b80-9ac7-11eb-909e-eda396fe4b9f.png)
![image](https://user-images.githubusercontent.com/57340671/114292077-0819a580-9ac7-11eb-846e-700aeee56d2d.png)

- About KNN Classification

The average accuracy of the cross-validation was 66 percent, which did not improve the accuracy compared to the previous KNN algorithm. Through these results, k-fold cross-validation can be used about small size data sets for evaluation and improve accuracy. Compared with this, about large size datasets, we realized that there is a disadvantage for that the calculation time is larger than the calculation by dividing train and test dataset.

![image](https://user-images.githubusercontent.com/57340671/114292081-1c5da280-9ac7-11eb-86e4-361d1534415e.png)
![image](https://user-images.githubusercontent.com/57340671/114292082-1e276600-9ac7-11eb-9f9e-f2a481089a1a.png)
![image](https://user-images.githubusercontent.com/57340671/114292085-1f589300-9ac7-11eb-982a-f23327ec39c5.png)
![image](https://user-images.githubusercontent.com/57340671/114292086-2089c000-9ac7-11eb-8cf6-c9df2985d131.png)
![image](https://user-images.githubusercontent.com/57340671/114292089-21baed00-9ac7-11eb-8db6-7fad5b3394a9.png)
![image](https://user-images.githubusercontent.com/57340671/114292092-22ec1a00-9ac7-11eb-89c7-1bd42c3cc6fb.png)
![image](https://user-images.githubusercontent.com/57340671/114292094-24b5dd80-9ac7-11eb-8f2b-8ffe98c33575.png)

### Ensemble Learning - Bagging
Because only classifier models can be used in the voting classification, bagging was performed with only KNN algorithm. As a result of bagging KNN, it was confirmed that the accuracy increased to 73%. it shows that bagging result accuracy is higher than 0.68 that is accuracy before bagging.

![image](https://user-images.githubusercontent.com/57340671/114292110-31d2cc80-9ac7-11eb-873c-7e5b44b22151.png)
![image](https://user-images.githubusercontent.com/57340671/114292111-339c9000-9ac7-11eb-9e2e-15519829cc5e.png)


***

## GUI
We created a GUI based on the code that we implemented, so that when the user enters it according to the column, we can predict the condition accordingly. 
First, set the size and tickle of the window and insert the text, input window, and button to go inside.

![image](https://user-images.githubusercontent.com/57340671/114292121-3e572500-9ac7-11eb-9492-4c7715fb2f53.png)
![image](https://user-images.githubusercontent.com/57340671/114292122-3f885200-9ac7-11eb-88a4-088b193b0c51.png)
![image](https://user-images.githubusercontent.com/57340671/114292124-40b97f00-9ac7-11eb-8cd9-7097f84f0a30.png)

Enter a value in the input window entry via the predict_multiple function and click button to convert it to labelEncoder if it is not a number and show the predicted condition value through the ensemble in gui.

![image](https://user-images.githubusercontent.com/57340671/114292129-4b741400-9ac7-11eb-902f-29a1de6bf423.png)
![image](https://user-images.githubusercontent.com/57340671/114292131-4c0caa80-9ac7-11eb-9ecf-f1e7ada928ab.png)


***
## Conclusion
Significant difference in prediction accuracy depending on data preprocessing and realized that preprocessing is very important.
In dealing with quite a lot of data, we experienced an unexpected error, the more systematic and meticulous data pre-processing is necessary.
it is necessary to apply appropriate algorithms according to the data, and all processes are important.
