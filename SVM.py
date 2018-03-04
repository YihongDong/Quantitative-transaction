from sklearn.model_selection import train_test_split
from read_by2 import *
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from math import log
import glob
import sklearn.preprocessing
import matplotlib.pyplot as plt

def error_percentage(train_result,y_train,weight):
    error = 0
    percentage_1 = 0
    percentage_0 = 0
    for i in range(len(train_result)):
        if y_train[i] == 1:
            percentage_1 += 1
            if train_result[i] != y_train[i]:
                error += weight[1] * 1
        else:
            percentage_0 += 1
            if train_result[i] != y_train[i]:
                error += weight[0] * 1
    error_percentage = error / (percentage_1 * weight[1] + percentage_0 * weight[0])
    return error_percentage

def am_coefficient(clf,x_train,y_train):
    weight = clf.class_weight_
    train_result = clf.predict(x_train)
    train_error_percentage=error_percentage(train_result,y_train,weight)
    a=1/2*log((1-train_error_percentage)/train_error_percentage)
    if a<0:
        a=0
    return a

if __name__ =="__main__":
    feature=[]
    label=[]
    for file in glob.glob("F:\\study in school\\量化交易\\1\\000005.csv"):
        feature1,label1 = process_execl(file)
        feature+=feature1[5:-1]
        label+=label1[5:-1]
    # label_change =list(map(numbers_to_feature,label[5:-1]))
    feature = sklearn.preprocessing.MinMaxScaler().fit_transform(feature)
    #feature = sklearn.preprocessing.normalize(feature)
    pca=PCA(n_components=40)
    feature=pca.fit_transform(feature)
    print(pca.explained_variance_)

    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2)

    clf1_rbf = SVC(kernel='rbf',probability=True,class_weight='balanced')
    clf1_rbf.fit(x_train, y_train)
    a1=am_coefficient(clf1_rbf,x_train,y_train)

    clf2_rbf = SVC(kernel='rbf',probability=True)
    clf2_rbf.fit(x_train, y_train)
    a2=am_coefficient(clf2_rbf,x_train,y_train)

    clf3_sigmoid = SVC(kernel='sigmoid', probability=True,class_weight='balanced')
    clf3_sigmoid.fit(x_train, y_train)
    a3=am_coefficient(clf3_sigmoid,x_train,y_train)

    clf4_sigmoid = SVC(kernel='sigmoid', probability=True)
    clf4_sigmoid.fit(x_train, y_train)
    a4=am_coefficient(clf4_sigmoid,x_train,y_train)

    a_total=a1+a2+a3+a4
    result=clf1_rbf.predict_proba(x_test)*a1+clf2_rbf.predict_proba(x_test)*a2\
           +clf3_sigmoid.predict_proba(x_test)*a3+clf4_sigmoid.predict_proba(x_test)*a4
    # print(result)
    result=result[:,1]
    real_reult=[]
    error=0
    error_1=0
    len_1=0
    for i in range(len(result)):
        if result[i]>a_total*0.5:
            real_reult.append(1)
            len_1+=1
            if y_test[i]!=1:
                error+=1
                error_1+=1
        else:
            real_reult.append(0)
            if y_test[i]!=0:
                error+=1

    print("rise recall:%f"%(error_1/len_1))
    print(error/len(result))
    print(y_test)
    print(real_reult)



    # x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2)
    # clf_rbf = SVC(kernel='rbf',probability=True,class_weight='balanced')
    # clf_rbf.fit(x_train, y_train)
    # weight=clf_rbf.class_weight_
    # print(clf_rbf.class_weight_)
    # print(len(x_train))
    # print(len(clf_rbf.support_vectors_))
    # train_result=clf_rbf.predict(x_train)
    # result=clf_rbf.predict(x_test)
    # print(y_test)
    # print(result)
    #
    # error= 0
    # percentage_1 = 0
    # for i in range(len(train_result)):
    #     if train_result[i]!=y_train[i]:
    #         error+=1
    #     if y_train[i]==1:
    #         percentage_1+=1
    # trainerror_percentage=error/len(train_result)
    # percentage_1=percentage_1/len(train_result)
    # print("training中1的比例为%f"%percentage_1)
    # print(trainerror_percentage)
    #
    # error = 0
    # percentage_1 = 0
    # for i in range(len(result)):
    #     if result[i]!=y_test[i]:
    #         if y_test[i]==0:
    #             error+=weight[0]*1
    #         else:
    #             error+=weight[1]*1
    #     if y_test[i]==1:
    #         percentage_1+=1
    # testerror_percentage=error/(percentage_1*weight[1]+(len(result)-percentage_1)*weight[0])
    # percentage_1 = percentage_1 / len(result)
    # print("test中1的比例为%f" % percentage_1)
    # print(testerror_percentage)
    #
    # clf_sigmoid = SVC(kernel='sigmoid',probability=True,class_weight=dict(zip([0,1],[1,1])))
    # clf_sigmoid.fit(x_train, y_train)
    # print(clf_rbf.class_weight_)
    # print(len(x_train))
    # print(len(clf_sigmoid.support_vectors_))
    # train_result = clf_sigmoid.predict(x_train)
    # result = clf_sigmoid.predict(x_test)
    # print(result)
    # print(y_test)
    # error = 0
    #
    # for i in range(len(train_result)):
    #     if train_result[i] != y_train[i]:
    #         error += 1
    # trainerror_percentage = error / len(train_result)
    # print(trainerror_percentage)
    #
    # error = 0
    # percentage_1 = 0
    # for i in range(len(result)):
    #     if result[i]!=y_test[i]:
    #         if y_test[i]==0:
    #             error+=weight[0]*1
    #         else:
    #             error+=weight[1]*1
    #     if y_test[i]==1:
    #         percentage_1+=1
    # testerror_percentage=error/(percentage_1*weight[1]+(len(result)-percentage_1)*weight[0])
    # percentage_1 = percentage_1 / len(result)
    # print("test中1的比例为%f" % percentage_1)
    # print(testerror_percentage)

