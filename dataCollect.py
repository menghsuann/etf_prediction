import csv
import pandas
import os
import codecs


allDataPath = "/Users/Allen/bigDataPlay/allData/2013_2018"
allData = os.listdir(allDataPath)

#c = open("allData.csv", "w")

f = codecs.open("allData.csv", 'wb', "Big5")
#writer = csv.writer(f)

writer = csv.writer(f)
writer.writerow(["日期","自營商買進金額","自營商賣出金額","自營商買賣差額", "投信買進金額","投信賣出金額","投信買賣差額","外資及陸資買進金額","外資及陸資賣出金額","外資及陸資買賣差額"])


def extractData(eachPath):
    print(eachPath)
    df = pandas.read_csv(eachPath, encoding= "Big5",skiprows=[0]) #read file
    rowDataList = []
    rowDataList.append(eachPath[-12:-4])
    for i in range (0 ,3):
        for j in range (1 ,4 ):
            rowDataList.append(df.iloc[i,j])
#print(rowDataList)
    writer.writerow(rowDataList)
  
def collectData ():
    
    for year in range (0 , len(allData)) :
        #print(allDataPath+"/"+allData[year])
        extractData( allDataPath+"/"+allData[year])

collectData()
f.close()


