from flask import Flask, render_template, request
from flask import jsonify
import csv
import numpy as np
import xgboost as xgb
import numpy as np
import pandas as pd
app = Flask(__name__)


# @app.route('/api', methods=['POST'])
# def api_handler():
#     # 在这里处理 /api 路径上的 POST 请求
#     return render_template('index 01.html')
#     pass

@app.route('/')
def index():
    return render_template('index 01.html')










@app.route('/submit', methods=['POST'])
def calculate():
    # 从表单中获取数据
    ph = float(request.form['土壤ph'])
    clay = float(request.form['土壤黏土含量'])
    cec = float(request.form['土壤CEC'])
    toc = float(request.form['TOC'])
    bet = float(request.form['BET'])
    temperature = float(request.form['温度'])
    solid_liquid_ratio = float(request.form['固液比'])
    equilibrium_concentration = float(request.form['溶液平衡浓度'])

    # 这边我进行了加和运算，你需要怎么算就怎么算吧
    np.random.seed(80)
    def readData_cv(K=9):
        df = pd.read_csv("C:\\data1\\new_data.csv")
        all_data = np.zeros((488, 10))
        all_data[:, 0] = df['CEC cmol/kg'].values
        all_data[:, 1] = df['pH'].values
        # Load 粘土含量 %
        all_data[:, 2] = df['TOC %'].values
        # Load 粘土含量 %
        all_data[:, 3] = df['粘土含量 %'].values
        # Load 粘土含量 %
        all_data[:, 4] = df['BET m2/g'].values
        # Load 温度
        all_data[:, 5] = df['温度'].values
        # Load 土壤/溶液 g/mL
        all_data[:, 6] = df['土壤/溶液 g/mL'].values
        # Load 溶液平衡浓度Ce ug/L
        all_data[:, 7] = df['溶液平衡浓度Ce ug/L'].values
        # Load log土壤上有机物吸附量 ug/g
        all_data[:, 8] = df['log土壤上有机物吸附量 ug/g'].values
        # Load isotherm
        all_data[:, 9] = df['isotherm'].values
        #Number of groups
        numOfGroup = np.zeros((1,))
        numOfGroup[0] = np.size(np.unique(all_data[:, 9]))
    #Cross validation split (K-fold)
    #k-1 fold, last fold as test set
    #Split according to group
        ratio = 1.0/K
        indice = []
        for i in range(1):
            index = np.arange(numOfGroup[i])
            np.random.shuffle(index)
            indice.append(index)
    #Split 
        groupID = np.unique(all_data[:, 9])
        data = [np.zeros((0, 9)) for _ in range(K)]
        data_len_normal = int(numOfGroup[0] * ratio)
        data_len_last = int(numOfGroup[0] - data_len_normal * (K - 1))
        for i in range(K):
            if(i != K-1):
                data_len = data_len_normal
            else:
                data_len = data_len_last
            for j in range(data_len):
                j += i * data_len_normal
                data[i] = np.append(data[i], all_data[all_data[:, 9] == groupID[int(indice[0][j])], 0:9], axis=0)
        cnt = 0
        for i in range(K):
            cnt += np.shape(data[i])[0]
        print(cnt, np.shape(all_data))
    
        X = []
        y = []
        for i in range(K):
            X.append(data[i][:, 0:8])
            y.append(data[i][:, 8])
    #X_folds (K-1 + 1)
        X_folds = []
        for i in range(K):
            X_folds.append(X[i])
    #y_folds
        y_folds = []
        for i in range(K):
            y_folds.append(y[i])
        cnt = 0
        for i in range(K):
            cnt += np.shape(X_folds[i])[0]
        print(cnt)
        return X_folds, y_folds



    np.random.seed(80)


    K = 9
    X_folds, y_folds = readData_cv(K)
    X_test = X_folds[K-1]
    y_test = y_folds[K-1]

    rmse_err = np.zeros((3, K-1))
    R2_err = np.zeros((3, K-1))
    mae_err = np.zeros((3, K-1))
    huber_err = np.zeros((3, K-1))
    logcosh_err = np.zeros((3, K-1))
    for i in range(K-1):
    #Generate training set and validation set
        X_valid = X_folds[i]
        y_valid = y_folds[i]
        X_train = np.zeros((0, 8))
        y_train = np.zeros((0,))
        for j in range(K-1):
            if(j != i):
                X_train = np.append(X_train, X_folds[j], axis=0)
                y_train = np.append(y_train, y_folds[j], axis=0)
        regr2 = xgb.XGBRegressor(n_estimators=50, min_child_weight=2)

        regr2.fit(X_train, y_train)


        y_pred_train = regr2.predict(X_train)
        y_pred_valid = regr2.predict(X_valid)
        y_pred_test = regr2.predict(X_test)



    x = pd.DataFrame([[cec,ph,toc,clay,bet,temperature,solid_liquid_ratio,equilibrium_concentration]], columns = ['f0','f1','f2','f3','f4','f5','f6','f7']) # 
    y = regr2.predict(x)
    #x =['pH':ph , '粘土含量':clay , 'CEC cmol/kg':cec , 'TOC %':toc , 'BET m2/g':bet , '温度':temperature , '土壤/溶液 g/mL':solid_liquid_ratio , '溶液平衡浓度Ce ug/L':equilibrium_concentration]
    result = y



    # 将结果返回给前端
    # return jsonify({'result': result})

    return render_template('result.html', result=result)



if __name__ == '__main__':
    app.run()
