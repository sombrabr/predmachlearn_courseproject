# Weight Lifting Exercise Recognition
Eduardo Bortoluzzi Junior  
May 27th, 2016  




## Abstract

This report analyses data collected from accelerometers on the belt, forearm, 
arm and dumbbell of 6 subjects doing weight lifting exercises to create a model
so it is possible to detect this type of exercise being doing in a wrong way
during the exercise series, not after the series is ended. 

The data has information when the subject was doing the exercise in the right 
way (class A), and when doing in 4 common wrong ways: throwing the elbows to the 
front (class B), lifting the dumbbell only halfway (class C), lowering the 
dumbbell only halfway (class D) and throwing the hips to the front (class E).

The final model could be applied to any subject at any time to detect in which
class the exercise is being done.

The model was created using machine learning in R language, with the *caret*
package.

More information about this data is available at [PUC RIO's Human Activity Recognition site](http://groupware.les.inf.puc-rio.br/har), at the section on
Weight Lifting Exercises Dataset.

## Model creation


The training data is loaded from [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv). It has 
19622 observations and 160 variables.

After analyzing the training data, it was detected that there are raw sensors
data and some derived data generated after aggregating those raw data for a time
period. As the aim of this model is to detect the class of the exercise
_during_, and not after, the exercise, the time series information and the
aggregated ones will be left out during the model training. The following
variables were left out:

* *X*: the order of the test, not relevant

* *user\_name*: the subject name, not relevant

* *raw\_timestamp\_part\_\**, *cvtd\_timestamp*: the time of the exercise series, 
  not relevant for the aim of the model, that is not a time series;
  
* *new\_window*, *num\_windows*: the aggregation window, not relevant;

* *kurtosis\_\**, *skewness\_\**, *max\_\**, *min\_\**, *amplitude\_\**, 
  *var\_\**, *avg\_\**, *stddev\_\**, *var\_\**: data generated after the
  aggregation in a time period, not relevant.


```r
to.remove = grepl("^(classe|X|user_name|raw_timestamp_part_|cvtd_timestamp|new_window|num_window|kurtosis_|skewness_|max_|min_|amplitude_|var_|avg_|stddev_|var_)", names(data.training))

# Training to predict "classe"
y = data.training[,c("classe")]
# Training data, without the removed ones
x = data.training[,c(!to.remove)]
```

The remaining variables are the roll (longitudinal axis), pitch (lateral axis),
yaw (vertical axis) rotations; total accelaration; and the x, y and z 
information of the gyroscope, accelerometer and magnetometer of each of the
sensors.

For high accuracy, Random Forest was selected as the predict function, using
10-fold cross validation. The cross validation was done inside the training
function, so the data was not split into the training and validation data sets.

With the Random Forest, several samples is bootstrapped from the training data
to grow different trees, so the best one can be selected.

This is the code for the training:



```r
set.seed(1234)
fitControl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)
modFit <- train(x, y, method="rf", data=data.training, trControl=fitControl)
```


The summary for the model is:


```
## Random Forest 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 17661, 17659, 17660, 17658, 17660, 17660, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9951073  0.9938106
##   27    0.9948528  0.9934889
##   52    0.9897566  0.9870409
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

And it is seen that the selected model has a high accuracy of 0.9951073, so no
further optimization will be done.

The low in sample error rate can be seen from the confusion matrix:


         A      B      C      D      E   class.error
---  -----  -----  -----  -----  -----  ------------
A     5578      1      0      0      1     0.0003584
B       10   3784      3      0      0     0.0034238
C        0     21   3400      1      0     0.0064290
D        0      0     41   3173      2     0.0133706
E        0      0      0      7   3600     0.0019407

## Applying the model to the test sample


The testing data is loaded from [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv). It has 
20 observations and 160 variables.

The prediction is done using the model created in the previous session, leaving
out the variables not important:


```r
pred = predict(modFit, data.testing[,!to.remove])
```

The classes of each exercise in the testing sample is not shown, but can be
generated with the code above.



## Conclusion

The Random Forest is not a fast function for prediction, as this prediction took
near 37 minutes in a Intel i3 
processor using some parallelism, but it produces great results, as seen in the 
model accuracy.

The probability of having the 20 classes of the testing samples
correctly predicted is
90.7%, a very good probability.
