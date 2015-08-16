enc = OneHotEncoder(n_values = np.array([8,8,27,6,5,9,10,5,69,61,8,25]))
chunk = pd.read_csv(trainpath3 + '0')
trainY = chunk["click"]
chunk = chunk[cols3]
vcd = enc.fit_transform(np.array(chunk)).toarray()
clf = RandomForestClassifier(n_estimators=5,n_jobs=-1)
clf.fit(vcd,trainY)


chunk2 = pd.read_csv(testfile2,usecols=cols4)
temptestpred = np.array([])
testpred = np.zeros(4577464) 
idlist = []
idlist.append([i for i in chunk2.id.tolist()])
chunk2 = chunk2[cols3]
tstvcd = enc.transform(np.array(chunk2)).toarray()
temp = clf.predict_proba(tstvcd)
temptestpred = np.concatenate((temptestpred,temp[:,1]))
testpred = testpred + temptestpred

idlist = sum(idlist,[])

fop = open(outputpath + 'submission.csv', 'w+')
fop.writelines('id,click\n')

for i in range(0,len(idlist)):
     fop.writelines(str(idlist[i]) + ',' + str(round(testpred[i],4)) + '\n')
 
fop.close()
clf.score(vcd,trainY)
