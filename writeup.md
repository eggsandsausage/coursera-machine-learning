
```
## Warning: package 'randomForest' was built under R version 3.1.1
```


Machine learning assignment: predict exercise classifications
========================================================

Goals of assignment

1. Predict the "classe" variable.

  a. Pick predictors
  
  b. Pick model
  
  c. Train model (use of cross validation)
  
2. Report out of sample error

## 1a: Picking the predictors
The test sets on which the final model was to be used on consists of 20 observations in a separate 20 by 160 matrix. However, out of these 160 variables, only 60 contains data. Since there would be no point in training a model on features that were not present in the testset, this was the first reduction of features. Furthermore, since the focus of the assignment was to predict classifications of exercise, I chose to remove the features that did not meassure actual sensor data related to lifting the weight (ie username, timestamps etc). 

Training the model on the remaining variables did not seem computationally taxing, and since the final test set was somewhat sparse I chose to use all of them to maintain as high an accuracy as possible. The variables used as predictors are listed below.


```
##  [1] "num_window"           "roll_belt"            "pitch_belt"          
##  [4] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
##  [7] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [10] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [13] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [16] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [19] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [22] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [25] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [28] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [31] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [34] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [37] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [40] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [43] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [46] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [49] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [52] "magnet_forearm_y"     "magnet_forearm_z"
```


## 1b: Picking the model
Two factors that were considered: accuracy and the large number of predictors. A large number of predictors could possibly lead to overfitting which would of course affect the final accuracy of the model. This could be countered by either reducing the number of predictors (either through PCA or some correlation analysis to find superfluous variables) or doing some kind of cross validation. However, since *random forests* provide both high accuracy and internal cross validation, this seemed like the way to go.

## 1c: Training the model:
When constructing trees in random forests, not all data is actually used in each iteration. About one third of the observations are out of bag (OOB) and are used for predicting an outcome of the constructed tree in a particular iteration. Since our model is trying to make classifications, the predictions will be summed and the majority prediction what the model finally delivers as its prediction. This is not entirely unlike cross validation, which also performs model construction over and over again, averaging out predictions into a final (hopefully) unbiased model. With large enough forrests, it can be shown that the OOB-error is virtually equivalent to the leave-one-out-cross-validation error. However, showing this might not be entirely within the scope of the assignment (plus it's getting kind of late), so below you'll find the results of a split into a training and test set, giving some indication of the out of sample error.


```r

intrain <- createDataPartition(y = tsubset$classe, p = 0.6, list = F)
trainingsub <- tsubset[intrain, ]
testsub <- tsubset[-intrain, ]
fitx <- randomForest(classe ~ ., data = trainingsub[, -c(1:6)], mtry = 8, ntree = 500)
pred1 <- predict(fitx, testsub[, -c(1:6)])
confusionMatrix(data = pred1, reference = testsub$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    1    0    0    0
##          B    1 1513    7    0    0
##          C    0    4 1361    8    0
##          D    0    0    0 1278    0
##          E    0    0    0    0 1442
## 
## Overall Statistics
##                                         
##                Accuracy : 0.997         
##                  95% CI : (0.996, 0.998)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.997         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.997    0.995    0.994    1.000
## Specificity             1.000    0.999    0.998    1.000    1.000
## Pos Pred Value          1.000    0.995    0.991    1.000    1.000
## Neg Pred Value          1.000    0.999    0.999    0.999    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.173    0.163    0.184
## Detection Prevalence    0.284    0.194    0.175    0.163    0.184
## Balanced Accuracy       1.000    0.998    0.997    0.997    1.000
```


## 2. Report out of sample error/conclusions
Looking at the results above the model looks like it's working nicely with a 0.997 accuracy within a narrow confidence interval. On a side note the OOB-error of the model is 0.0037, stressing the previous point that cross validation was probably a superfluous step in this particular model construction.
