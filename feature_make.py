# TODO:特徴量を製作するためのプログラム(train、test両データに対応)

#　必要なライブラリのインポート
import glob
import pandas as pd
import csv
from geopy.distance import vincenty
import re
from statistics import mean, median,variance,stdev

# path
make_csvdir = "feature_csv/"

# trianの中にあるcsvデータを全読み込みする
train_files = glob.glob('abc2018dataset/train/*.csv')

# testの特徴量も抽出する必要がある。
test_files = glob.glob('abc2018dataset/test/*.csv') # testの中にあるcsvデータを読み込む

# NOTE: trainとtestで使い分ける
#ext_data,savefilename = train_files,'train_feature.csv';
ext_data,savefilename = test_files,'test_feature.csv';

#1つのcsvデータの中身を抜き取理、特徴量csvを作成(ヘッダーはデータ詳細の略称をつける)
"""
※headerの対応表
lon : longitude
lat : latitude
sun_azimuth : sun azimuth [degree] clockwise from the North
sun_elevation : sun elevation [degree] upward from the horizon
day_night_time : (1) daytime (between sunrise and sunset), or (0) nighttime
elapsed_time : elapsed time [second] after starting the trip
local_time : local time (hh:mm:ss)
days : days (starts from 0, and increments by 1 when the local time passes 23:59:59)
"""

# 抽出する特徴量のヘッダー名
feature_list = [["csv_no","sum_d(m)","v_mean","v_median","v_variance","v_stdev","all_elapsed_t","oneday_d_mean","one_day_d_median"
,"dt_par","nt_par","sun_e_max","sun_e_min","sun_a_max","length_dis","side_dis","len_dis_ratio","si_dis_ratio","sun_a_variance","sun_a_median","sun_a_inter","sun_e_minus_data_variance","sun_e_per","v_max","state_ratio","lat_mean","lon_mean","v_non_stay_mean","v_non_stay_median","v_non_stay_variance","v_non_stay_stdev","sun_a_mean","sun_e_mean","sun_a_variance","sun_e_variance",'lon_variance','lat_variance',"lon_median","lat_median","lon_max","lat_max","lon_min","lat_min","tugaru_cnt","day_info","season","start_time","end_time"]]
# 特徴量の配列を用意する

for i in range(len(ext_data)):
#for i in range(1): #デバッグ用
    data = pd.read_csv(ext_data[i],header=None)
    my_dict = {"lon":data[0],"lat":data[1],"sun_azimuth":data[2],"sun_elevation":data[3],"day_night_time":data[4],"elapsed_time":data[5],"local_time":data[6],"days":data[7]}
    df = pd.DataFrame.from_dict(my_dict)

    # 使用する変数
    dis = [] # 移動距離用の配列
    e_times = [] # 経過時間の行の差
    v_all = [] # 速度の配列
    one_d = 0 #1日の移動距離を保管する変数
    one_day_d = [] #1日あたりの総移動距離
    dt0_cnt = 0 #夜の時間をcountする変数
    nt1_cnt = 0 #昼の時間をcountする変数
    sun_e_max = 0 #太陽高度の最大値
    sun_e_min = 10000 #太陽高度の最小値
    sun_rise_set = []
    sun_a_max = 0
    lon_max = 0
    lon_min = 1000
    lat_max = 0
    lat_min = 1000
    sun_a_per = []
    sun_a_inter = [] #sun_aの周期の平均値を求める
    sun_a_in_val = [] #sun_a_perに値を周期ごとに保存するための保管配列
    sun_e_minus_data = [] #グラフ内で低下している時の値を取得する
    day_info = 0 #データ取得日数
    v_max = 0
    tugaru_cnt = 0 # 津軽海峡（41.63250000, 140.40527777）を通ったかをcount

    for j in range(len(df)-1):
        # 総移動距離の計算
        distance = vincenty((df["lat"][j], df["lon"][j]),(df["lat"][j+1], df["lon"][j+1])).meters #m単位にする
        str_d = str(distance) #型を文字列に変換
        #print(str_d)
        ext_d = re.search(r'(\d+\.\d+)', str_d) #文字列を距離のみを抽出
        d_float = float(ext_d.group(1))
        dis.append(d_float)
        # 隣接距離の統計処理用、速度[m/s]
        e_time = df["elapsed_time"][j] - df["elapsed_time"][j+1]
        e_times.append(abs(e_time)) #時間差分の絶対値をリストに保存
        v = dis[j] / e_times[j]
        v_all.append(v) # 速度(v_all)に対して統計処理を行い、feature_listに挿入

        #1日あたりの総移動距離
        one_d += d_float
        if df["days"][j] != df["days"][j+1]:
            one_day_d.append(one_d)
            one_d = 0

        #昼夜の割合を調べるためにcountする
        if df["day_night_time"][j] == 0:
            dt0_cnt += 1
        else:
            nt1_cnt +=1

        #太陽高度の最大値を調べる
        if sun_e_max < df["sun_elevation"][j]:
            sun_e_max = df["sun_elevation"][j]

        #太陽高度の最小値を調べる
        if sun_e_min > df["sun_elevation"][j]:
            sun_e_min = df["sun_elevation"][j]

        if sun_a_max < df["sun_azimuth"][j]:
            sun_a_max = df["sun_azimuth"][j]

        # lonの最大最小値を求める
        if lon_max < df["lon"][j]:
            lon_max = df["lon"][j]
        if lon_min > df["lon"][j]:
            lon_min = df["lon"][j]
        # latの最大最小値を求める
        if lat_max < df["lat"][j]:
            lat_max = df["lat"][j]
        if lat_min > df["lat"][j]:
            lat_min = df["lat"][j]

        if df["lat"][j] >= 40 or df["lon"][j] >= 140:
            tugaru_cnt += 1

        # sun_azimuthの高度の下がる間隔に違いがあるように見える
        sun_a_in_val.append(df["sun_azimuth"][j])
        if abs(df["sun_azimuth"][j] - df["sun_azimuth"][j+1]) >= 100: #次の列との差が100以上であれば周期が変わったとみなす
            sun_a_per.append(sun_a_in_val)
            sun_a_in_val = []
        elif df["sun_azimuth"][j+1] == df["sun_azimuth"][len(df["sun_azimuth"])-1]:
            sun_a_per.append(sun_a_in_val)

        #sun_elevationが下がって行く回数を数える
        if df["sun_elevation"][j+1] - df["sun_elevation"][j] < 0:
            sun_e_minus_data.append(df["sun_elevation"][j])

    bf_lt1 = df["local_time"][0]
    #print(bf_lt1)
    af_lt1 = re.search('(\d+):(\d+):(\d+)', bf_lt1)
    s_hour = int(af_lt1.group(1))
    s_minu = int(af_lt1.group(2))
    s_sec = int(af_lt1.group(3))

    start_time = (s_hour * 60 * 60) + (s_minu * 60) + s_sec

    bf_lt2 = df["local_time"][len(df["local_time"])-1]
    #print(bf_lt2)
    af_lt2 = re.search('(\d+):(\d+):(\d+)', bf_lt2)
    e_hour = int(af_lt2.group(1))
    e_minu = int(af_lt2.group(2))
    e_sec = int(af_lt2.group(3))

    end_time = (e_hour * 60 * 60) + (e_minu * 60) + e_sec
    #print(end_time)

    if not sun_e_minus_data:
        sun_e_minus_data.append(0)
        sun_e_minus_data.append(0)

    day_info = (df["days"][len(df["days"])-1]) +1 #データ取得日数
    sun_e_per = len(sun_e_minus_data) / day_info #マイナスの数を日にちで割る

    if not one_day_d:
        one_day_d.append(one_d) #1日しか移動していない場合を配列に格納

    for k in range(len(sun_a_per)):
        sun_a_inter.append(len(sun_a_per[k])) #周期ごとののデータ数

    #昼夜の割合の計算
    dt_par = (dt0_cnt / (dt0_cnt+nt1_cnt)) * 100 #昼間の割合
    nt_par = (dt0_cnt / (nt1_cnt+nt1_cnt)) * 100 #昼間の割合

    length_dis = vincenty((lat_min,lon_min),(lat_max,lon_min)).meters
    side_dis = vincenty((lat_min,lon_min),(lat_min,lon_max)).meters
    len_dis_ratio = length_dis / (length_dis + side_dis)
    si_dis_ratio = side_dis / (length_dis + side_dis) #移動距離の縦横

    sum_d = sum(dis) # 総移動距離
    all_elapsed_t = df["elapsed_time"][len(df["elapsed_time"])-1] #総経過時間

    for l in range(len(v_all)):
        if v_max < v_all[l]:
            v_max = v_all[l]

    v_stay = [x for x in v_all if x < 1]
    state_ratio = len(v_stay)/len(v_all)

    v_non_stay = [x for x in v_all if x > 1]

    # 抽出する特徴量のヘッダー名 にlat_meanとlon_meanを追加
    lat_mean = mean(df["lat"])
    lon_mean = mean(df["lon"])
    sun_a_mean = mean(df["sun_azimuth"])
    sun_e_mean = mean(df["sun_elevation"])
    sun_a_variance = variance(df["sun_azimuth"])
    sun_e_variance = variance(df["sun_elevation"])

    # 季節の情報
    if sun_e_max >= 55:
        season = 1
    else:
        season = 0

    # 特徴量をcsvリストに入れる
    feature_list.append([i,sum_d,mean(v_all),median(v_all),variance(v_all),stdev(v_all),all_elapsed_t,mean(one_day_d),median(one_day_d),round(dt_par,2),round(nt_par,2),float(sun_e_max),float(sun_e_min),sun_a_max,length_dis,side_dis,len_dis_ratio,si_dis_ratio,variance(list(df["sun_azimuth"])),median(list(df["sun_azimuth"])),mean(sun_a_inter),variance(sun_e_minus_data),sun_e_per,v_max,state_ratio,lat_mean,lon_mean,mean(v_non_stay),median(v_non_stay),variance(v_non_stay),stdev(v_non_stay),sun_a_mean,sun_e_mean,sun_a_variance,sun_e_variance,variance(df["lon"]),variance(df["lat"]),median(df["lon"]),median(df["lat"]),lon_max,lat_max,lon_min,lat_min,tugaru_cnt,day_info,season,start_time,end_time])

    print("No."+str(i)+" end")


# 特徴量をcsvにして出力する
with open(make_csvdir + savefilename, 'w') as f:
    writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
    writer.writerows(feature_list)
