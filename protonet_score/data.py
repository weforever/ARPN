import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
# 读取和预处理数据
def load_data():
  data_u = pd.read_csv(r'D:/protonet_1/data_zhinengche.csv')
  data_use = data_u.loc[:,['车速', '累计里程', '总电压', '总电流', 'SOC', 'DC-DC状态', '挡位', '挡位驱动力', '挡位制动力',
            '最高电压电池单体代号', '电池单体电压最高值', '最低电压电池单体代号', '电池单体电压最低值',
            '最高温度探针单体代号', '最高温度值', '最低温度探针子系统代号', '最低温度值',
            '可充电储能装置故障总数']]

  X = data_use.copy()
  y = X.pop('可充电储能装置故障总数')

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

  X_train = np.asarray(X_train)
  X_test = np.asarray(X_test)
  y_train = np.asarray(y_train)
  y_test = np.asarray(y_test)

  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.fit_transform(X_test)

  # 返回预处理后的数据
  return X_train_scaled, y_train, X_test_scaled, y_test