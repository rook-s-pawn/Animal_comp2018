# TODO:ランダムフォレストを使ってモデルを作るためのプログラム

# 必要なライブラリをインポート
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

# dataとlabelの呼び出し

name = ['sum_d(m)','v_max','oneday_d_mean','one_day_d_median','lat_mean','lon_mean','v_non_stay_mean','v_non_stay_median','v_non_stay_variance','lon_variance','lat_variance','lon_median','lat_median']
data = pd.read_csv("feature_csv/train_feature.csv", header=0)
data = data[name]
label = pd.read_csv("abc2018dataset/train_labels.csv", header=None)

"""
print("\n+ labelのヘッダ :\n")
print(label.head(5))
print("\n+ train_dataのヘッダ :\n")
print(data.head(5))
"""

# trainデータとtestデータに分ける
test_rate=0.01 #trainとtestの分ける割合
data_train, data_test, label_train, label_test = cross_validation.train_test_split(data, label,test_size=test_rate, random_state=40)


#クロスバリデーションの数
n_cv = 10

#学習機
mod = RandomForestClassifier()


#パラメータ(RF)
parameters = {
    'n_estimators' : [300],#[150,200,300],#[300],#[150], # test_rate=0.05の時 150で93%
    'max_depth'    : [30,50],#[10,30,50],#
    'random_state' : [58],#[7], # test_rate=0.05の時 58で93%
    'criterion' :['gini', 'entropy'],
}
"""
#パラメータ(svm)
parameters = [
#{"C": [3000,5000],"kernel":["linear"]},
{"C": [3000,5000],"kernel":["rbf"],"gamma":[0.001,0.0001]}
]
"""

clf = GridSearchCV(mod, parameters,cv=n_cv,n_jobs=-1)
clf.fit(data_train, label_train.as_matrix().reshape(-1,))

#クロスバリデーション
print("各正解率 = ",clf.grid_scores_)

print("\n+ ベストパラメータ（グリッドサーチで見つけた最適値）:\n")
print(clf.best_estimator_)

# Feature_importances
print("\n+ 各特徴量の重要度:\n")
fti = clf.best_estimator_.feature_importances_
for i, feat in enumerate(data.columns):
    print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))

#データを予測
predict = clf.predict(data_test)

#モデルのシリアライズ
joblib.dump(clf, 'predict_file/model_rf.pkl')
print("シリアライズOK")

#あっているか結果を確認
ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test, predict)
print("\n+ 正解率 :\n")
print("ac_score : ", ac_score)
print("\n+ レポート :\n")
print("cl_report : \n", cl_report)

# コンフュージョンマトリックス（混同行列）
import my_func as mf
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Compute confusion matrix
cnf_matrix = confusion_matrix(label_test, predict)#, labels=["e","p"])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
mf.plot_confusion_matrix(cnf_matrix, classes=["0","1"],
                     title='Confusion matrix, without normalization')
plt.savefig("fig/"+ "mf_without_normalization.png")

# Plot normalized confusion matrix
plt.figure()
mf.plot_confusion_matrix(cnf_matrix, classes=["0","1"], normalize=True,
                     title='Normalized confusion matrix')
plt.savefig("fig/"+ "mf_Normalized_confusion.png")

print('Best cross-validation: {}'.format(clf.best_score_))
