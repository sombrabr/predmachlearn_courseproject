# Download the data
TRAINING_FILE = "pml-training.csv"
TESTING_FILE = "pml-testing.csv"

if(! file.exists(TRAINING_FILE)) {
  download.file(
    url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
    destfile = TRAINING_FILE)
}

if(! file.exists(TESTING_FILE)) {
  download.file(
    url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
    destfile = TESTING_FILE)
}

# Get some lines from the file
conn.training = file(TRAINING_FILE, "r")
lines = readLines(conn.training, n=10)
print(lines)

# Read as CSV
csv.training = read.csv(TRAINING_FILE)
csv.testing = read.csv(TESTING_FILE)

# Get some information about the training data
names(csv.training)
summary(csv.training)
str(csv.training)

# Function to detect columns with too many empty strings
invalid.column = function(x) {
  if(is.factor(x)) {
    return((length(which(x=="")) / length(x)) > 0.5)
  } else {
    return(FALSE)
  }
}

# Detect the columns to remove due lack of data
to.remove = sapply(csv.training, invalid.column)
print(names(csv.training[,to.remove]))

to.remove.right = grepl("^(classe|X|user_name|raw_timestamp_part_|cvtd_timestamp|new_window|num_windows|kurtosis_|skewness_|max_|min_|amplitude_|var_|avg_|stddev_|var_)", names(csv.training))

summary(csv.training[,!to.remove.right])

# Train
library(caret)
y = csv.training[,c("classe")]
x = csv.training[,c(!to.remove.right)]

library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv",number = 10,allowParallel = TRUE)

modFit <- train(x,y, method="rf",data=csv.training,trControl=fitControl)

stopCluster(cluster)

p = predict(modFit, csv.testing[,!to.remove.right])
modFit$results[modFit$results$mtry==modFit$bestTune$mtry,c("Accuracy")]
