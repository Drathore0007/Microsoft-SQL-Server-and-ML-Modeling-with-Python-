CREATE PROCEDURE ML_py_predict (@model varchar(100)) 

AS 

BEGIN 

  DECLARE @py_model varbinary(max) = (select model from ML_py_models where model_name = @model); 

  EXEC sp_execute_external_script 

        @language = N'Python', 

        @script = N' 

import pickle 

from pandas import read_csv, DataFrame 

from sklearn.linear_model import LinearRegression  

from sklearn import preprocessing, cross_validation 

import numpy as np 

 

data = read_csv("E:\\amazon_data.csv") 

forcast = int(10) 

data = data[["Adj. Close"]] 

data["prediction"] = data[["Adj. Close"]].shift(-forcast) 

feature = np.array(data.drop(["prediction"],1)) 

X = preprocessing.scale(feature) 

forcast_value = X[-forcast:] 

X= X[:-forcast] 

lr = pickle.loads(py_model) 

predictions = lr.predict(forcast_value) 

df = DataFrame({"Amazon_Predicted_stock": predictions}) 

OutputDataSet = df 

print(df)' 

, @input_data_1 = N'' 

, @input_data_1_name = N'' 

, @params = N'@py_model varbinary(max)' 

, @py_model = @py_model 

with result sets (("Amazon_Predicted_stock" float)); 

END; 

GO 