def weight_func(pred,weight,func):
    """
    function to calculate weighted predictions
    """
    prediction = []
    for i in range(len(pred)):
        not_nan = np.logical_not(np.isnan(pred[i,:]))
        current_weight = weight[not_nan]
        current_pred = pred[i,not_nan]
        current_weight = func(current_weight)
        prediction.append(np.dot(current_weight,current_pred.T))
    return prediction

def ensemble_selective(learners,X_train,X_test,y_train,y_test,ttest_size,ltol,stol,k,metric):
    """
    function using several machine learners to do selective learning
    'learners': list of supervised machine learners
    'X_train','X_test','y_train','y_test': training and testing data
    'ttest_size': propotion of train_test data in train dataset
    'ltol': tolerance for select learners
    'stol': tolerance for select bad samples
    'k': number of nearset neighbors of bad samples to find in test data and train_train data
        bad samples are these samples performed bad in train_test data of reg
    'metric': distance metric for finding nearest neighbors of bad samples
    """
    from sklearn.neighbors import KDTree
    X_ttrain,X_ttest,y_ttrain,y_ttest = train_test_split(X_train,y_train,test_size=ttest_size,random_state=42)
    predictions = pd.DataFrame()
    weights = []
    ensemble_mse = {}
    result = {}
    for learner in learners:
        learner.fit(X_ttrain,y_ttrain)
        learner_pred = learner.predict(X_ttest)
        select_score = MSE(y_ttest,learner_pred)
        if select_score < ltol:
            # find index of bad samples
            bottom_idx = list(y_ttest[(abs(learner_pred-y_ttest) >= stol)].index)
            bad_samples = X_ttest.loc[bottom_idx,:]
            # find the indices of nearest neighbors of bad samples in training and testing data
            tree = KDTree(X_ttrain,metric=metric)
            _,train_drop_ind = tree.query(bad_samples,k=k)
            train_drop_ind = sum(train_drop_ind.tolist(),[])
            train_drop_ind = X_ttrain.index[train_drop_ind]
            tree = KDTree(X_test,metric=metric)
            _,test_drop_ind = tree.query(bad_samples,k=k)
            test_drop_ind = sum(test_drop_ind.tolist(),[])
            test_drop_ind = X_test.index[test_drop_ind]
    
            # clean testing data
            test_drop_ind = test_drop_ind.unique()
            clean_test_data = X_test.drop(test_drop_ind,axis=0)
            clean_test_label = y_test.drop(test_drop_ind,axis=0)
            # clean training data
            train_drop_ind = train_drop_ind.unique()
            clean_train_data = X_train.drop(train_drop_ind,axis=0)
            clean_train_data = clean_train_data.drop(bottom_idx,axis=0)
            clean_train_label = y_train.drop(train_drop_ind,axis=0)
            clean_train_label = clean_train_label.drop(bottom_idx,axis=0)
            
            # refit classifier using clean_training data
            learner.fit(clean_train_data,clean_train_label)
            # make predictions on clean testing data
            selective_pred = learner.predict(clean_test_data)
            
            name = learner.__class__.__name__
            selective_pred = pd.DataFrame(selective_pred,index=clean_test_data.index,columns=[name])
            predictions = predictions.join(selective_pred,how='outer')
            weights.append(select_score)
            
    
    ens_mean = predictions.apply(np.nanmean,axis=1)
    ensemble_mse['mean'] = MSE(y_test.loc[predictions.index,],ens_mean)
    ens_median = predictions.apply(np.nanmedian,axis=1)
    ensemble_mse['median'] = MSE(y_test.loc[predictions.index,],ens_median)
    array_pred = predictions.as_matrix()
    weights = np.array(weights)
    func1 = lambda w : (1./w**2)/sum(1./w**2)
    inversed_mse_pred = weight_func(array_pred,weights,func1)
    ensemble_mse['inversed_mse'] = MSE(y_test.loc[predictions.index,],inversed_mse_pred)
    func2 = lambda w : w/sum(w)
    mse_pred = weight_func(array_pred,weights,func2)
    ensemble_mse['mse'] = MSE(y_test.loc[predictions.index,],mse_pred)
    func3 = lambda w : np.exp(-w)/sum(np.exp(-w))
    softmax_mse_pred = weight_func(array_pred,weights,func3)
    ensemble_mse['softmax_mse'] = MSE(y_test.loc[predictions.index,],softmax_mse_pred)
    
    drop_proportion = 1.-len(predictions)/len(y_test)
    
    result['ensemble_mse'] = ensemble_mse
    result['drop_proportion'] = drop_proportion
    result['selected_learner'] = list(predictions.columns)
    
    return result