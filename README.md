# selective_learning

The main idea of selective learning is feature engineering. It tries to clean training data and even testing data for machine learning models to improve their performance.

It contains following several steps:

1.Split the original dataset into training and testing set and further split training dataset into train_train and train_test.

2.Fit machine learning model using train_train dataset and make predictions on train_test dataset

3.Measure the performance of model on train_test dataset according to some performance measures and mark those samples that model performed bad on them as bad samples, which means this model might not suitable for these data.

4.
