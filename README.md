# Forcasting-and-Visualization-Of-Stocks.
Stock market is an important leading indicator of where the economy will be in the future. Stock market is volatile and more dynamic so there are various news media available
for spreading ideas through online investing platforms which leads to strong concoction of market dynamics. 
However, Machine Learning algorithms are used for analyzing and predicting the rise and fall of the stock market. 
The stock market is considered be to more volatile as there are various factors for deciding the stock value in the market.
The project mainly focuses on the use of Long Short-Term Memory based Machine learning to predict stock market values.
Various factors are considered such as open, close, high, low and volume to predict stock market price of next 30 days.
Stock market prediction is a complex problem as there are many factors that have yet to be addressed by developers and researchers.
With the proper use of machine learning methods,we can easily relate previous data to the current data and also train the machine to learn and make accurate assumptions.

--------------DATA COLLECTION--------------
The dataset must be concrete as minor changes in the data can result in massive differences. In this paper,NSE PY library is used to extract historical and real time data. 
This dataset consists of the following variables: open, close, low, high and volume and adjacent close. 
Open, close, low and high are bid prices for the stock at separate times. Volume is the number of shares which are passed from one owner to another.
Also MinMaxScaler is used to scale the chosen features in a required range. 
---------------------Working----------------
[Forcasting and Visualization Of Stocks Folder](https://github.com/abhang16/Forcasting-and-Visualization-Of-Stocks/Nifty-Prediction)
Contains following Django Project files and folders.
└───Stock_Prediction
└───lstm
└───stock
|   db.sqlite3
|   manage.py
|   nifty50Companies.csv 


- #### [lstm](https://github.com/abhang16/Forcasting-and-Visualization-Of-Stocks/Nifty-Prediction/Nifty-50-Prediction/lstm)
  > It is special kind of recurrent neural network that is capable of learning long term dependencies in data. 
  This is achieved because the recurring module of the model has a combination of four layers interacting with each other.
  > lstm Directory
 ```
 lstm
 |   RunModel.py
 |   TrainModel.py
 |   lstmModel_final.json
 
 
 -----------------------------------------------Prediction for 30 days------------------------------------------
following code is from [RunModel.py](https://github.com/abhang16/Forcasting-and-Visualization-Of-Stocks/Nifty-Prediction/Nifty-50-Prediction/lstm/RunModel.py)
```python
def getNext30Days(self):
        self.__inputHandler()
        dataset = self.data
        dataset = dataset['Close'].values
        dataset = dataset[len(dataset)-30:]
        n_features = 1
        n_steps = 30
        past_days = 30
        # demonstrate prediction for next 30 days
        x_input = np.array(dataset.tolist())
        temp_input = list(x_input)
        lst_output = []
        i = 0
        while(i < 30):

            if(len(temp_input) > past_days):
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape((1, n_steps, n_features))
                yhat = self.model.predict(x_input, verbose=0)
                temp_input.append(yhat[0][0])
                temp_input = temp_input[1:]
                lst_output.append(yhat[0][0])
                i = i+1
            else:
                x_input = x_input.reshape((1, n_steps, n_features))
                yhat = self.model.predict(x_input, verbose=0)
                temp_input.append(yhat[0][0])
                lst_output.append(yhat[0][0])
                i = i+1
        print(lst_output)
        predictions = lst_output
        return predictions
```
  - #### [stock](https://github.com/abhang16/Forcasting-and-Visualization-Of-Stocks/Nifty-Prediction/Nifty-50-Prediction/stock)
  > A Django application is a Python package that is specifically intended for use in a Django project. 
  An application may use common Django conventions, such as having models , tests , urls , and views submodules.
  > stock Directory
```
stock
└───migrations
└───templates
|   |   base.html
|   |   home.html
|   |   signup.html
|   |   aboutus.html
|   |   fundamental.html
|   |   technical.html
|   __init__.py
|   admin.py
|   apps.py
|   forms.py
|   models.py
|   tests.py
|   views.py
```
  - #### [db.sqlite3](https://github.com/abhang16/Forcasting-and-Visualization-Of-Stocks/Nifty-Prediction/Nifty-50-Prediction/db.sqlite)
  > SQLite3 is a software library that provides a relational database management system. The lite in SQLite means lightweight in terms of setup, 
  database administration, and required resources. SQLite has the following noticeable features: self-contained, serverless, zero-configuration, transactional.
  > We are using sqlite3 for manageing User Authentication


  - #### [manage.py](https://github.com/abhang16/Forcasting-and-Visualization-Of-Stocks/Nifty-Prediction/Nifty-50-Prediction/manage.py)
  > A command-line utility that lets you interact with this Django project in various ways. 
  You can read all the details about manage.py in django-admin and manage.py. The inner mysite/ directory is the actual Python package for your project.
  - #### [nifty50Companies.csv](https://github.com/abhang16/Forcasting-and-Visualization-Of-Stocks/Nifty-Prediction/Nifty-50-Prediction/nifty50Companies.csv)
  > Csv file containing the list of nifty 50 companies with their respective symbol
### 3. [models Folder](https://github.com/abhang16/Forcasting-and-Visualization-Of-Stocks/Nifty-Prediction/Nifty-50-Prediction/models)
Contains experiments with models 
  > models Directory
  ```
  models
  |   NiftyPrediction.ipynb
  |   NiftyPrediction.ipynb

 |   weights_final.h5
 
 
 
 
 
