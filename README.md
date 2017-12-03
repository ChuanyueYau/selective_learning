# Machine Learning
## Ensemble Selective Learning System

This project employs several supervised algorithms using selective learning and ensemble 'good' learners' predictions to predict implied volatility (continous) of Options. The main idea of selective learning is feature engineering. It tries to clean training data and even testing data for machine learning models to improve their performance.
The steps of selective learning are:
It contains following several steps:
- 1.Split the original dataset into training and testing set and further split training dataset into train_train and train_test.
- 2.Fit machine learning model using train_train dataset and make predictions on train_test dataset
- 3.Measure the performance of model on train_test dataset according to some performance measures and mark those samples that model performed bad on them as 'bad' samples, which means this model might not suitable for these data.
- 4.Find the K (e.g.K=10) nearest neighbors of 'bad' samples in train_train and test dataset and remove these neighbors, the assumption is the learning machine might also performs worse on neighbors of 'bad' samples. This is the process of cleaning training and testing data.
- 5.Re-train the learning machine using the clean training data and make predictions on clean testing data.

The ensemble system will drop some bad-performed learners, then weights the predictions of selected learners (every learners are trained on its own clean_training data and make predictions on its own clean_testing data) using several weighting methods and generate new prediction results.


### Install

This project requires **Python 3.5** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)


### Code

The code is provided in the `selective_learning.py` python file. You will also be required to include the `NBOption.csv` dataset file to run the code. 

### Run

In a terminal or command window, navigate to the top-level project directory `selective_learning/` (that contains this README) and run one of the following commands:

```bash
python3.5 selective_learning.py
```  
