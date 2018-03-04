import csv
import numpy as np
def process_execl(path):
    csv_reader = csv.reader(open(path, encoding='utf-8'))
    getfeature =[]
    k=1
    i=1
    kidfeature =[]
    label = []
    key =True
    data=list(csv_reader)
    for row in data[1:]:
        row=list(map(float,row))
        if(k!=48):
            if k!=row[1]:
                key = False
                k=row[1]
            kidfeature.append([((row[2]-data[i-1][3])/data[i-1][3])if type(data[i-1][3])==float else 0\
                                  ,(row[3]-row[2])/row[2],(row[4]-row[2])/row[2],(row[5]-row[2])/row[2]])
            kidfeature.append(row[6:])
            k=k+1
        else:
            if (k != row[1]):
                key = False
                k = row[1]
            if key ==True:
                kidfeature.append([((row[2] - data[i - 1][3]) / data[i - 1][3]) if type(data[i - 1][2]) == float else 0 \
                                      , (row[3] - row[2]) / row[2], (row[4] - row[2]) / row[2],(row[5] - row[2]) / row[2]])
                kidfeature.append(row[6:])
                getfeature.append(sum(kidfeature,[]))
                tem = float(data[i + 1][2]) if i<len(data)-1 else 0
                # if (tem - row[3]) / row[3] >= 0.002:
                #     label.append(1)
                # elif (tem - row[3]) / row[3] <= -0.002:
                #     label.append(-1)
                # else:
                #     label.append(0)
                if (tem - row[3]) / row[3] >0:
                    label.append(1)
                else:
                    label.append(0)
            kidfeature=[]
            k = 1
            key=True
        i=i+1
    return getfeature,label




