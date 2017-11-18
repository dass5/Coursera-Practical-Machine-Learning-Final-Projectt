Coursera Practical Machine learning final project
The Dataset:
The data comes from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.
The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set.

Load the library:
library(caret)

Read the dataset in R:
trainUrl <- http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
testUrl <- http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
training <- read.csv(url(trainUrl))
testing <- read.csv(url(testUrl))
dim(training)
The training dataset consists of 19622 rows and 160 columns .Let's see what these Columns are:
colnames(training)
[1] "X"                        "user_name"                "raw_timestamp_part_1"     "raw_timestamp_part_2"    
  [5] "cvtd_timestamp"           "new_window"               "num_window"               "roll_belt"               
  [9] "pitch_belt"               "yaw_belt"                 "total_accel_belt"         "kurtosis_roll_belt"      
 [13] "kurtosis_picth_belt"      "kurtosis_yaw_belt"        "skewness_roll_belt"       "skewness_roll_belt.1"    
 [17] "skewness_yaw_belt"        "max_roll_belt"            "max_picth_belt"           "max_yaw_belt"            
 [21] "min_roll_belt"            "min_pitch_belt"           "min_yaw_belt"             "amplitude_roll_belt"     
 [25] "amplitude_pitch_belt"     "amplitude_yaw_belt"       "var_total_accel_belt"     "avg_roll_belt"           
 [29] "stddev_roll_belt"         "var_roll_belt"            "avg_pitch_belt"           "stddev_pitch_belt"       
 [33] "var_pitch_belt"           "avg_yaw_belt"             "stddev_yaw_belt"          "var_yaw_belt"            
 [37] "gyros_belt_x"             "gyros_belt_y"             "gyros_belt_z"             "accel_belt_x"            
 [41] "accel_belt_y"             "accel_belt_z"             "magnet_belt_x"            "magnet_belt_y"           
 [45] "magnet_belt_z"            "roll_arm"                 "pitch_arm"                "yaw_arm"                 
 [49] "total_accel_arm"          "var_accel_arm"            "avg_roll_arm"             "stddev_roll_arm"         
 [53] "var_roll_arm"             "avg_pitch_arm"            "stddev_pitch_arm"         "var_pitch_arm"           
 [57] "avg_yaw_arm"              "stddev_yaw_arm"           "var_yaw_arm"              "gyros_arm_x"             
 [61] "gyros_arm_y"              "gyros_arm_z"              "accel_arm_x"              "accel_arm_y"             
 [65] "accel_arm_z"              "magnet_arm_x"             "magnet_arm_y"             "magnet_arm_z"            
 [69] "kurtosis_roll_arm"        "kurtosis_picth_arm"       "kurtosis_yaw_arm"         "skewness_roll_arm"       
 [73] "skewness_pitch_arm"       "skewness_yaw_arm"         "max_roll_arm"             "max_picth_arm"           
 [77] "max_yaw_arm"              "min_roll_arm"             "min_pitch_arm"            "min_yaw_arm"             
 [81] "amplitude_roll_arm"       "amplitude_pitch_arm"      "amplitude_yaw_arm"        "roll_dumbbell"           
 [85] "pitch_dumbbell"           "yaw_dumbbell"             "kurtosis_roll_dumbbell"   "kurtosis_picth_dumbbell" 
 [89] "kurtosis_yaw_dumbbell"    "skewness_roll_dumbbell"   "skewness_pitch_dumbbell"  "skewness_yaw_dumbbell"   
 [93] "max_roll_dumbbell"        "max_picth_dumbbell"       "max_yaw_dumbbell"         "min_roll_dumbbell"       
 [97] "min_pitch_dumbbell"       "min_yaw_dumbbell"         "amplitude_roll_dumbbell"  "amplitude_pitch_dumbbell"
[101] "amplitude_yaw_dumbbell"   "total_accel_dumbbell"     "var_accel_dumbbell"       "avg_roll_dumbbell"       
[105] "stddev_roll_dumbbell"     "var_roll_dumbbell"        "avg_pitch_dumbbell"       "stddev_pitch_dumbbell"   
[109] "var_pitch_dumbbell"       "avg_yaw_dumbbell"         "stddev_yaw_dumbbell"      "var_yaw_dumbbell"        
[113] "gyros_dumbbell_x"         "gyros_dumbbell_y"         "gyros_dumbbell_z"         "accel_dumbbell_x"        
[117] "accel_dumbbell_y"         "accel_dumbbell_z"         "magnet_dumbbell_x"        "magnet_dumbbell_y"       
[121] "magnet_dumbbell_z"        "roll_forearm"             "pitch_forearm"            "yaw_forearm"             
[125] "kurtosis_roll_forearm"    "kurtosis_picth_forearm"   "kurtosis_yaw_forearm"     "skewness_roll_forearm"   
[129] "skewness_pitch_forearm"   "skewness_yaw_forearm"     "max_roll_forearm"         "max_picth_forearm"       
[133] "max_yaw_forearm"          "min_roll_forearm"         "min_pitch_forearm"        "min_yaw_forearm"         
[137] "amplitude_roll_forearm"   "amplitude_pitch_forearm"  "amplitude_yaw_forearm"    "total_accel_forearm"     
[141] "var_accel_forearm"        "avg_roll_forearm"         "stddev_roll_forearm"      "var_roll_forearm"        
[145] "avg_pitch_forearm"        "stddev_pitch_forearm"     "var_pitch_forearm"        "avg_yaw_forearm"         
[149] "stddev_yaw_forearm"       "var_yaw_forearm"          "gyros_forearm_x"          "gyros_forearm_y"         
[153] "gyros_forearm_z"          "accel_forearm_x"          "accel_forearm_y"          "accel_forearm_z"         
[157] "magnet_forearm_x"         "magnet_forearm_y"         "magnet_forearm_z"         "classe"    

So there are 160 columns all together. Our dependent variable is “classe”. Let's see how many of them are A, how many B and C,D or E from the data in training
table(training$classe)
A    B    C    D    E 
5580 3797 3422 3216 3607 

Data Cleaning:
Missing Data:
Let’s take a look at what columns have missing values:
summary(training)
nacols=training[colSums(is.na(training))>0]
dim(nacols)
colSums(is.na(nacols))

It looks like we have missing values in the several columns (67). All the 67 columns have 19216 missing values out of 19622 records. We will not get any meaningful insight from these columns.  So we can safely remove these columns. In general, if we have more than 60% of NA values in a particular column,We will drop the column
Remove the columns which have more than 60% NA values
train<-training[!colSums(is.na(training))>(nrow(training)*.6)]
dim(train) 

[1] 19622    93

Zero Variance:
Zero and near-zero variance predictors happen quite often across samples. Zero variance means datasets come with predictors that take a unique value across samples. For many models (excluding tree-based models), this may cause the model to crash or the fit to be unstable.
To remove predictors like those nearZeroVar from the caret package is used. It not only removes predictors that have one unique value across samples (zero variance predictors), but also removes predictors that have large ratio of the frequency of the most common value to the frequency of the second most common value (near-zero variance predictors).
Remove zero variance or nearly zero variance columns
badCols <- nearZeroVar(train)
train <- train[, -badCols] 
dim(train)
[1] 19622    59

We can see columns x,user_name,raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp of no use. We can ignore these
train<-train[,-c(1,2,3,4,5)]
Now we are down to 54 columns

Identify Correlated Predictors:
While there are some models that thrive on correlated predictors (such as pls), other models may benefit from reducing the level of correlation between the predictors. 
Given a correlation matrix, the findCorrelation function uses the following algorithm to flag predictors for removal: 
#Check for corelated variables(dont consider Classe variable)
descrCor <- cor(train[,c(1:53)])
# Check for abolutely corelated variables.There are none.
highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > 0.999)

We set cut off is .75. If more than that we will remove those variables
highlyCorDescr <- findCorrelation(descrCor, cutoff = 0.75)
# Remove highly co related variables
train <- train[, -highlyCorDescr]
dim(train)
dim(train)
[1] 19622    33

Prediction Algorithms:

Create some models of the data and estimate their accuracy on unseen data:
The steps followed:
1. Separate out a validation dataset.
2. Set-up the test harness to use 5-fold cross validation.
3. Build 5 different models to predict classe variable
4. Select the best model.

Create validation dataset using Data Split:
We split the dataset into two, 70% of will be used to train the models and 30% that will be held back as a validation dataset. The validation set will be used to get a second and independent idea of how accurate the best model might actually be. It will also give us out of sample error
This gives more concrete estimate of the accuracy of the best model on unseen data by evaluating it on actual unseen data.
intrain<-createDataPartition(y=train$classe,p=0.7,list=FALSE)
train_final <- train[intrain,]
test_final <- train[-intrain,]

Test Harness:
5-fold cross validation is used to estimate accuracy.
This will split our dataset into 5 parts, train on 4 and test on 1 and repeat for all combinations of train-test splits.
#Create the model
control <- trainControl(method="repeatedcv", number=5)
The metric “Accuracy” is used to evaluate models. This is a ratio of the number of correctly predicted instances in divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). The out of sample will be calculated as (1-accuracy) multiplied by 100 to give a percentage.
Build Models:
Spot check 5 different algorithms to check wich one gives better accuracy on training dataset:
1. Stochastic Gradient Boosting (Generalized Boosted Modeling GBM)
2. K-Nearest Neighbors (KNN).
3. Classification and Regression Trees (CART).
4. Gaussian Naive Bayes (NB).
5. Random Forest(RF).
The random number seed is set before each run to ensure that the evaluation of each algorithm is performed using exactly the same data splits. It ensures the results are directly comparable.
seed<-123
## Stochastic Gradient Boosting (Generalized Boosted Modeling)
set.seed(seed)
fit.gbm <- train(classe~., data=train_final, method="gbm",  preProc=c("center", "scale"), trControl=control)
# K nearesr neighbor algorithm
set.seed(seed)
fit.knn <- train(classe~., data=train_final, method="knn",  trControl=control)
# Naive Bayes
set.seed(seed)
fit.nb <- train(classe~., data=train_final, method="nb",  trControl=control)
# CART
set.seed(seed)
fit.cart <- train(classe~., data=train_final, method="rpart", trControl=control)
# Random Forest
set.seed(seed)
fit.rf<- train(classe~., data=train_final, method="rf",  trControl=control)

Select the best model:
We now have 5 models and accuracy estimations for each. We need to compare the models to each other and select the most accurate.
results <- resamples(list(GBM=fit.gbm, KNN=fit.knn, NaiveBayes=fit.nb, CART=fit.cart, RandomForest=fit.rf))

#summarize the result of the algorithms as a table.
summary(results)

Call:
summary.resamples(object = results)

Models: GBM, KNN, NaiveBayes, CART, RandomForest 
Number of resamples: 5 

Accuracy 
               Min. 1st Qu. Median   Mean 3rd Qu.   Max. NA's
GBM          0.9873  0.9894 0.9895 0.9897  0.9909 0.9916    0
KNN          0.8487  0.8519 0.8554 0.8550  0.8581 0.8609    0
NaiveBayes   0.7337  0.7429 0.7444 0.7435  0.7471 0.7496    0
CART         0.5575  0.5662 0.5765 0.5745  0.5795 0.5929    0
RandomForest 0.9960  0.9967 0.9967 0.9972  0.9975 0.9989    0

Kappa 
               Min. 1st Qu. Median   Mean 3rd Qu.   Max. NA's
GBM          0.9839  0.9867 0.9867 0.9870  0.9885 0.9894    0
KNN          0.8087  0.8128 0.8172 0.8166  0.8203 0.8241    0
NaiveBayes   0.6662  0.6773 0.6789 0.6781  0.6826 0.6853    0
CART         0.4395  0.4512 0.4647 0.4621  0.4690 0.4859    0
RandomForest 0.9949  0.9959 0.9959 0.9964  0.9968 0.9986    0

Let’s review the results using a few different visualization techniques to get an idea of the mean and spread of accuracies.
# boxplot comparison
bwplot(results)
 
# Dot-plot comparison
dotplot(results)
 
Now we can see Random Forest and GBM algorithm are the most accurate models that we tested. Now we want to get an idea of the accuracy of the model on our validation set.
This will give us an independent final check on the accuracy and out of sample error of the best model. It is valuable to keep a validation set just in case you made a slip during training, such as overfitting to the training set or a data leak. Both will result in an overly optimistic result.

Let’s check for variable importance:
varImp(fit.rf)
plot(varImp(fit.rf))
 
We can run the Random Forest and GBM model directly on the validation set and summarize the results as a final accuracy score, a confusion matrix and a classification report and out of sample error.
Predictions:
# Make prediction in the validation dataset for RF model
predictions <- predict(fit.rf, test_final,type="raw")
table(predictions)
table(predictions$classe)
confusionMatrix(predictions, test_final$classe) 
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1671    3    0    0    0
         B    3 1134    3    0    0
         C    0    2 1023    0    0
         D    0    0    0  963    1
         E    0    0    0    1 1081

Overall Statistics
                                          
               Accuracy : 0.9978          
                 95% CI : (0.9962, 0.9988)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9972          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9982   0.9956   0.9971   0.9990   0.9991
Specificity            0.9993   0.9987   0.9996   0.9998   0.9998
Pos Pred Value         0.9982   0.9947   0.9980   0.9990   0.9991
Neg Pred Value         0.9993   0.9989   0.9994   0.9998   0.9998
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2839   0.1927   0.1738   0.1636   0.1837
Detection Prevalence   0.2845   0.1937   0.1742   0.1638   0.1839
Balanced Accuracy      0.9987   0.9972   0.9983   0.9994   0.9994

# Make prediction in the validation dataset for GBM model
predictions <- predict(fit.gbm, test_final,type="raw")
table(predictions)
confusionMatrix(predictions, test_final$classe) 

Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1672   10    0    0    0
         B    2 1117    9    1    1
         C    0    8 1015    4    0
         D    0    4    2  957    9
         E    0    0    0    2 1072

Overall Statistics
                                          
               Accuracy : 0.9912          
                 95% CI : (0.9884, 0.9934)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9888          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9988   0.9807   0.9893   0.9927   0.9908
Specificity            0.9976   0.9973   0.9975   0.9970   0.9996
Pos Pred Value         0.9941   0.9885   0.9883   0.9846   0.9981
Neg Pred Value         0.9995   0.9954   0.9977   0.9986   0.9979
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2841   0.1898   0.1725   0.1626   0.1822
Detection Prevalence   0.2858   0.1920   0.1745   0.1652   0.1825
Balanced Accuracy      0.9982   0.9890   0.9934   0.9948   0.9952

We can see that Random forest gives slightly better performance than GBM. The Random Forest algorithm gives an accuracy of 0.9978 or 99.78%. The out of sample error is .0022 or .22%.The confusion matrix provides an indication of the ten errors made. 
Predictions on testing set:
We’re now ready to fit the model to our test data and make our predictions.
Prepare the test dataset:
#Remove the classe variable from train_final
train_class<-train_final[,-c(55)]

# Allow only variables in testing that are also in train_final which is used to create the model
colnames<-colnames(train_class)
test_df_final<-testing[colnames]

#Make predictions
predict(fit.rf , test_df_final)

B A B A A E D B A A B C B A E E A B B B

