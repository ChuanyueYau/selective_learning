# selective_learning

The main idea of selective learning is feature engineering. It tries to clean training data and even testing data for machine learning models to improve their performance.

It contains following several steps:

1.Split the original dataset into training and testing set and further split training dataset into train_train and train_test.

2.Fit machine learning model using train_train dataset and make predictions on train_test dataset

3.Measure the performance of model on train_test dataset according to some performance measures and mark those samples that model performed bad on them as 'bad' samples, which means this model might not suitable for these data.

4.Find the K (e.g.K=10) nearest neighbors of 'bad' samples in train_train and test dataset and remove these neighbors, the assumption is the learning machine might also performs worse on neighbors of 'bad' samples. This is the process of cleaning training and testing data.

5.Re-train the learning machine using the clean training data and make predictions on clean testing data.
