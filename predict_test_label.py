# TODO:モデルを使ってテストのラベルを作成するためのプログラム

# 必要なライブラリをインポート
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics, svm
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
import os

# testデータの特徴量ファイルを読み込む
data_dir = "feature_csv/"
data_file = data_dir + 'test_feature.csv'

# 保存用ディレクトリがなかったら作成
if not os.path.exists('result_label/'):
   os.mkdir('result_label/')

#データを読み込む
name = ['sum_d(m)','v_max','oneday_d_mean','one_day_d_median','lat_mean','lon_mean','v_non_stay_mean','v_non_stay_median','v_non_stay_variance','lon_variance','lat_variance','lon_median','lat_median']

data = pd.read_csv(data_file , header=0)
data = data[name]
#モデルの読み込み
clf = joblib.load('predict_file/model_rf.pkl')

# ラベルを予測する
pred_label = clf.predict(data)

print("len(data) = ", len(data))
print("pred_label = ", pred_label)

pd.Series(pred_label).to_csv('result_label/y_submission.txt', index=False)
