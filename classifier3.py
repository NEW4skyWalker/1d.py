import time
from sklearn import svm
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score, confusion_matrix)
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



#第三章
def eth_label(s):
    it = {b'eth': 0, b'noneth': 1}
    return it[s]

def tcp_classifier():
    # 第三章读取数据集

    path = 'ch3_tcp_features.txt'
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={79: eth_label})
    data[np.isinf(data)]=np.nan
    data[np.isnan(data)]=0

    # 2.划分数据与标签
    x_old, y = np.split(data, indices_or_sections=(79,), axis=1)  # x为数据，y为标签
    x_old = x_old[:, 0:79]
    skb = SelectKBest(mutual_info_classif, k=12)  # 特征选择
    x_train1 = skb.fit_transform(x_old, y)
    x_test1 = skb.transform(x_old)
    select_name_index = skb.get_support(indices=True)
    print(select_name_index)

    x = SelectKBest(mutual_info_classif, k=12).fit_transform(x_old, y.ravel())
    print('new x shape is ',x.shape)
    x=x[:,0:11]

    start = time.perf_counter()
    train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.8,
                                                                       test_size=0.2)  # sklearn.model_selection.

    #RFmodel generate
    clf = RandomForestClassifier()
    clf.fit(train_data, train_label.ravel())
    end = time.perf_counter()
    print("RF time:", end - start, '\n')
    #joblib.dump(clf_RF, 'RFmodel.pkl')

    #clf = joblib.load('RFmodel.pkl')
    test_labelP=clf.predict(test_data)

    print("RF")

    print("训练集：", clf.score(train_data, train_label))

    print("测试集：", clf.score(test_data, test_label))
    print('RF report is ',classification_report(test_label, test_labelP))

    print("\n\n\n")

        #Logistics
    clf_LT = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial')
    clf_LT.fit(train_data, train_label.ravel())
    start = time.perf_counter()
    y_lt_pred = clf_LT.predict(test_data)

    print("logstics")
    print(sklearn.metrics.classification_report(test_label, y_lt_pred, digits=8))
    print("训练集：", clf_LT.score(train_data, train_label))

    print("测试集：", clf_LT.score(test_data, test_label))
    print('Lg report is ',classification_report(test_label, y_lt_pred,digits=8))

    end = time.perf_counter()
    print("Logistics time:", end - start)

    #3.训练svm分类器

    classifier = svm.SVC()
    classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先

    y_pred = classifier.predict(test_data)

    # SVM
    start = time.perf_counter()
    print("SVM:")
    print("训练集：", classifier.score(train_data, train_label))
    print("测试集：", classifier.score(test_data, test_label))

    print('SVM report is ',classification_report(test_label, y_pred,digits=8))

    end = time.perf_counter()
    print("SVM time:", end - start)
    print("\n\n\n")

    #4.KNN
    clf=KNeighborsClassifier()
    clf.fit(train_data,train_label.ravel())
    y_pred = clf.predict(test_data)
    print("训练集：", clf.score(train_data, train_label))
    print("测试集：", clf.score(test_data, test_label))
    print('KNN report is ',classification_report(test_label, y_pred,digits=8))

def udp_classifier():
    # 第三章读取数据集

    path = 'ch3_udp_features.txt'
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={77: eth_label})
    data[np.isinf(data)]=np.nan
    data[np.isnan(data)]=0

    # 2.划分数据与标签
    x_old, y = np.split(data, indices_or_sections=(77,), axis=1)  # x为数据，y为标签
    x_old = x_old[:, 0:77]
    skb = SelectKBest(mutual_info_classif, k=12)  # 特征选择
    x_train1 = skb.fit_transform(x_old, y)
    x_test1 = skb.transform(x_old)
    select_name_index = skb.get_support(indices=True)
    print(select_name_index)

    x = SelectKBest(mutual_info_classif, k=12).fit_transform(x_old, y.ravel())
    print('new x shape is ',x.shape)
    x=x[:,0:11]

    print('new x shape is ',x.shape)
    start = time.perf_counter()

    train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.8,
                                                                      test_size=0.2)  # sklearn.model_selection.

    #RFmodel generate
    clf_RF = RandomForestClassifier()
    clf_RF.fit(train_data, train_label.ravel())

    end = time.perf_counter()
    print("RF time:", end - start, '\n')
    # joblib.dump(clf_RF, 'RF2model.pkl')

    # clf = joblib.load('RF2model.pkl')
    test_labelP = clf_RF.predict(test_data)

    print("RF")
    print("训练集：", clf_RF.score(train_data, train_label))
    print("测试集：", clf_RF.score(test_data, test_label))
    print('RF report is ',classification_report(test_label, test_labelP,digits=8))


    #Logistics
    clf_LT = LogisticRegression()
    clf_LT.fit(train_data, train_label.ravel())
    start = time.perf_counter()
    y_lt_pred = clf_LT.predict(test_data)

    print("训练集：", clf_LT.score(train_data, train_label))
    print("测试集：", clf_LT.score(test_data, test_label))
    print('Lg report is ',classification_report(test_label, y_lt_pred,digits=8))
    end = time.perf_counter()
    print("Logistics time:", end - start)


    #3.训练svm分类器
    clfsvm = svm.SVC()
    clfsvm.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先
    y_svmpred = clfsvm.predict(test_data)

    # SVM
    start = time.perf_counter()
    print("SVM:")
    print("训练集：", clfsvm.score(train_data, train_label))
    print("测试集：", clfsvm.score(test_data, test_label))
    print('SVM report is ',classification_report(test_label, y_svmpred,digits=8))
    end = time.perf_counter()
    print("SVM time:", end - start)
    print("\n\n\n")

    #4.KNN
    clf=KNeighborsClassifier()
    clf.fit(train_data,train_label.ravel())
    y_pred = clf.predict(test_data)
    print("训练集：", clf.score(train_data, train_label))
    print("测试集：", clf.score(test_data, test_label))
    print('KNN report is ',classification_report(test_label, y_pred,digits=8))


udp_classifier()