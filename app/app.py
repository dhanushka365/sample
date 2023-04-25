from flask import Flask,render_template
import json
import socket
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import tensorflow as tf
from keras.layers import Dropout, RepeatVector, TimeDistributed
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route("/train")
def train_model():
    np.random.seed(1)


    #Load the sequence data from csv
    #I:\DM PROJECT\originaldataset\AEP_hourly.csv
    df = pd.read_csv('I:\DM PROJECT\originaldataset\AEP_hourly.csv')
    requests = pd.DataFrame(data=df,columns=['AEP_MW'])
    #if any missing value fill it by previous value and convert all requests into integer type
    requests.ffill(inplace=True)
    requests["AEP_MW"]=requests["AEP_MW"].astype(float).astype(int)

    dataset = df
    dataset["Month"] = pd.to_datetime(df["Datetime"]).dt.month
    dataset["Year"] = pd.to_datetime(df["Datetime"]).dt.year
    dataset["Date"] = pd.to_datetime(df["Datetime"]).dt.date
    dataset["Time"] = pd.to_datetime(df["Datetime"]).dt.time
    dataset["Week"] = pd.to_datetime(df["Datetime"]).dt.isocalendar().week
    dataset["Day"] = pd.to_datetime(df["Datetime"]).dt.day_name()
    dataset = df.set_index("Datetime")
    dataset.index = pd.to_datetime(dataset.index)

    df.head()
    df.isnull().sum()

    df2 = df.dropna()
    df2.head()

    #scale the data
    scaler = StandardScaler()
    scaled_requests = scaler.fit_transform(requests)

    #Traing data has to be sequential
    train_size = int(len(df)*0.80)
    test_size =len(df)-train_size

    #Number of samples to lookback for each sample
    #720 default
    lookback =720

    #sperate training and test data
    train_data = scaled_requests[0:train_size,:]

    #Add an additional week for lookback
    test_data = scaled_requests[train_size:len(df),:1]
    ##########################################################################################################
    #Build a LSTM model with Keras
    ##########################################################################################################
    #pepare RNN Dataset
    def create_rnn_dataset(data, lookback=1):
    
        data_x,data_y = [],[]
        for i in range(len(data)- lookback -1):
            a = data[i:(i + lookback),0]
            data_x.append(a)
            data_y.append(data[i + lookback,0])
        return np.array(data_x),np.array(data_y)

    #create x and y for training
    x_train , y_train = create_rnn_dataset(train_data , lookback)

    #Reshape for use with LSTM
    x_train = np.reshape(x_train,(x_train.shape[0],1,x_train.shape[1]))

    print("shapes of x,y:",x_train.shape , y_train.shape)
    ########################################################################################################

    tf.random.set_seed(21)
    ts_model =  Sequential()
    ts_model.add(LSTM(64,return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
    ts_model.add(LSTM(64,return_sequences=True))
    ts_model.add(LSTM(64,return_sequences=True))
    ts_model.add(LSTM(64))
    ts_model.add(Dense(1))
    ts_model.compile(loss="mean_squared_error",optimizer="adam")
    ts_model.summary()
    history = ts_model.fit(x_train, y_train, epochs=2,validation_split=0.1, batch_size=1500,verbose=1)

    ############################################################################################################
    #Test the Model
    #############################################################################################################
    #Preprocess
    test_req_x , test_req_y =create_rnn_dataset(test_data , lookback)
    test_req_x = np.reshape(test_req_x,(test_req_x.shape[0],1,test_req_x.shape[1]))
    ts_model.evaluate(test_req_x , test_req_y, verbose=1)

    #predict for the training dataset
    predict_on_train = ts_model.predict(x_train)
    #Prdeict on the test dataset
    predict_on_test = ts_model.predict(test_req_x)

    #train_mae_loss = np.mean(np.abs(predict_on_train - x_train), axis=1)

    ##############################################################################################################
    #accuracy score
    ############################################################################################################
    from sklearn import metrics
    import os
    score = np.sqrt(metrics.mean_squared_error(predict_on_test,test_req_y))
    print(f'After training the score is:{score}')
    #########################################################################################################
    predict_on_train = scaler.inverse_transform(predict_on_train)
    predict_on_test = scaler.inverse_transform(predict_on_test)

    ###############################################################################################################
    os.chdir(r'app\model_save')
    os.getcwd()
    ts_model.save(os.path.join(os.getcwd(),"lstm_model.h5"))
    return "Model train sucessfully"

@app.route("/login")
def login():
    return "login sucessfull"

@app.route("/")
def index():
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        return render_template('index.html', hostname=host_name, ip=host_ip)
    except:
        return render_template('error.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
