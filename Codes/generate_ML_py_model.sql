
CREATE PROCEDURE generate_ML_py_model (@trained_model varbinary(max) OUTPUT) 

    AS 

    BEGIN 

    EXECUTE sp_execute_external_script 

    @language = N'Python' 

    , @script = N' 

    from pandas import read_csv, DataFrame 

    from sklearn.linear_model import LinearRegression  

    #import matplotlib.pyplot as plt 

    #import quandl 

    from sklearn import preprocessing, cross_validation 

    import numpy as np 

    import pickle 

    data = read_csv("E:\\amazon_data.csv") 

    forcast = int(10) 

    data = data[["Adj. Close"]] 

    data["prediction"] = data[["Adj. Close"]].shift(-forcast) 

    feature = np.array(data.drop(["prediction"],1)) 

    #feature = np.array(data["Adj. Close"]) 

    label = np.array(data["prediction"]) 

    X = preprocessing.scale(feature) 

    y = label[:-forcast] 

    forcast_value = X[-forcast:] 

    X= X[:-forcast] 

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size =0.2) 

    lr = LinearRegression() 

    lr.fit(X_train, y_train) 

    #Before saving the model to the DB table, we need to convert it to a binary object 

    trained_model = pickle.dumps(lr)' 

    , @input_data_1 = N'' 

    , @input_data_1_name = N'' 

    , @params = N'@trained_model varbinary(max) OUTPUT' 

    , @trained_model = @trained_model OUTPUT; 

    END; 

    GO 