import sys
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def fileReading(path):
    data = (pd.read_csv(path, header=None,skiprows=[1]).values)
    # print(data)
    typeDictionary =  {}
    numColumns = data.shape[1]
    numRows = data.shape[0]
    for i in range(numColumns-1):
        datatype = (data[0,i]).split(":")[1]
        if(datatype=="Continuous"):
            typeDictionary[i] = 1
        else:
            typeDictionary[i]=0
    # print(typeDictionary)
    # print(len(typeDictionary)) 
        # while(True):
    x = np.ones((numRows-1,numColumns-1),dtype=int)
    y = np.ones((numRows-1,1),dtype=int)
    for i in range(1,numRows):
        x[i-1] = data[i,0:numColumns-1]
        y[i-1] = data[i,numColumns-1:numColumns]
    # print(data.shape)
    return((x,y),typeDictionary)


# def makeTree(Data):
def chooseBestAttribute(x,y,dictionary):
    # for the given data find the best attribute 
    # by finding the attribute with max MI and return it
    
    # 1. Find H(y)
    m = len(y)

    probabY = [0.0]*7
    for i in range(m):
        index = (int)(y[i][0]-1)
        probabY[index] = probabY[index] + 1
    H_y = 0.0

    for i in range(7):
        probabY[i]= probabY[i]/m
        if(probabY[i]!=0):
            H_y = H_y + probabY[i]*(math.log((1/probabY[i]),2))
    
    numAttr = len(dictionary)
    # print("inside here Hy  "+str(H_y))
    
    # 2. Then find H(Y/X) for each attribute
    ret =  -1
    maxval = 0
    # print(dictionary)
    for i in range(numAttr):
        if(dictionary[i]==1):#continuous case
            # first find median 
            # then split on it
            # remaining parts remain same
            listValues = []
            for k in range(m):
                listValues.append(x[k][i])
            listValues.sort()
            medianIndex = 0
            if(m%2 ==0):
                medianIndex = m//2 - 1
            else:
                medianIndex = m//2
            medianval = listValues[medianIndex]

            
            num0 = 0
            py0 = [0.0]*7
            num1 = 0
            py1 = [0.0]*7
            for j in range(len(x)):
                if(x[j][i]<= medianval):
                    num0 = num0 + 1
                    py0[y[j][0]-1]=py0[y[j][0]-1]+1
                if(x[j][i]> medianval):
                    num1 = num1 +1
                    py1[y[j][0]-1]=py1[y[j][0]-1]+1
            # if(m == num0 + num1):
            #     print("total points matching continuous")
            temp0 = 0.0
            temp1=0.0
            for k in range(7):
                if(num0!=0):
                    py0[k] = py0[k]/num0
                if(py0[k]!=0):
                    temp0 = temp0+ (py0[k]*(math.log((1/py0[k]),2)))
                if(num1!=0):
                    py1[k] = py1[k]/num1
                if(py1[k]!=0):
                    temp1 =  temp1 + (py1[k]*(math.log((1/py1[k]),2)))
            H_yx = (num0*temp0/m)+(num1*temp1/m)
            MI_yx= H_y-H_yx
            if(maxval < MI_yx):
                maxval = MI_yx
                ret = i

        else:# discrete case
            # print("into diszret with i value "+str(i))
            # value = 0
            num0 = 0
            py0 = [0.0]*7
            num1 = 0
            py1 = [0.0]*7
            for j in range(len(x)):
                # print(x[j][i])
                if(x[j][i]== 0):
                    num0 = num0 + 1
                    py0[y[j][0]-1]=py0[y[j][0]-1]+1
                elif(x[j][i]==1):
                    num1 = num1 +1
                    py1[y[j][0]-1]=py1[y[j][0]-1]+1
            # if(m == num0 + num1):
            #     print("total points matching discrete")
            temp0 = 0.0
            temp1=0.0
            for k in range(7):
                if(num0!=0):
                    py0[k] = py0[k]/num0
                if(py0[k]!=0):
                    temp0 = temp0+ (py0[k]*(math.log((1/py0[k]),2)))
                if(num1!=0):
                    py1[k] = py1[k]/num1
                if(py1[k]!=0):
                    temp1 =  temp1 + (py1[k]*(math.log((1/py1[k]),2)))
            H_yx = (num0*temp0/m)+(num1*temp1/m)
            MI_yx= H_y-H_yx
            # print("hyx  "+str(H_yx))
            # print("mi yx  "+str(MI_yx))
            if(maxval < MI_yx):
                maxval = MI_yx
                ret = i
    return (ret,maxval)

class treeNode:
    def __init__(self,parent,children=[],index= -1,continuous= False,medianValue = -1,leaf=False, leafValue =-1, total = 0, highest = 0,hval= 0.0):
        self.parent = parent
        self.children = children
        self.index = index
        self.continuous = continuous
        self.medianValue = medianValue
        self.leaf = leaf
        self.leafValue = leafValue
        self.total = total
        self.highest = highest
        self.hval = hval


def growTree(x,y,dictionary,parent,currDepth,maxDepth):
    if(currDepth>=maxDepth):
        numPoints = len(y)
        ans =0 
        # donnot handle this case now...
        # if numPoints ==0 :
        #     ret = treeNode(parent, [], -1, False,-1, True, 0)
        #     return ret
        if numPoints>=1:
            ally = [0]*7
            for i in range(numPoints):
                ally[y[i][0]-1]=ally[y[i][0]-1]+1
            output = np.argmax(ally)
            ret = treeNode(parent, [], -1, False,-1, True, output)
            return ret

    numPoints = len(y)
    ans =0 
    # if numPoints ==0 :
    #     ret = treeNode(parent, [], -1, False,-1, True, 0)
    #     return ret
    if numPoints>=1:
        for i in range(numPoints):
            if y[i][0] == y[0][0]:
                ans = ans +1
        if(ans == numPoints):
            # make a leaf node with label as y[0] and return 
            # ret = treeNode(parent,[],)
            # print("Comes new Here in all pure value= "+str(y[0][0]) + " ans is "+ str(ans))
            ret = treeNode(parent, [], -1, False,-1, True, y[0][0]-1)
            return ret
    (j,gain) = chooseBestAttribute(x,y,dictionary)
    # make new node with j index and continuity based on the node
    if(gain<=1e-4):
        # print("Comes Here in gaiin<=0")
        y_values = [0]*7
        for i in range(numPoints):
            y_values[y[i][0]-1]=y_values[y[i][0]-1]+1
        # make a leaf node with label as y[0] and return 
        # ret = treeNode(parent,[],)
        outputValue = np.argmax(y_values)
        # print("leaf node at depth "+str(currDepth)+" final y value is "+str(output))
        ret = treeNode(parent, [], -1, False,-1, True, outputValue)
        return ret
    isContinuous = False
    # print("j value is "+str(j))
    if(dictionary[j]==1):
        isContinuous = True
    else:
        isContinuous= False
    # print("non-leaf node at depth "+str(currDepth)+" attribute split is "+str(j))
    y_values = [0]*7
    for i in range(numPoints):
        y_values[y[i][0]-1]=y_values[y[i][0]-1]+1
    # make a leaf node with label as y[0] and return 
    # ret = treeNode(parent,[],)
    outputValue = np.argmax(y_values)
    
    nodej = treeNode(parent,[],j,isContinuous,-1,False,outputValue)
    if(isContinuous):
        listValues = []
        m = len(x)
        for k in range(m):
            listValues.append(x[k][j])
        listValues.sort()
        medianIndex = 0
        if(m%2 ==0):
            medianIndex = m//2 - 1
        else:
            medianIndex = m//2
        medianval = listValues[medianIndex]
        # handle last
        x0=[]
        y0=[]
        x1=[]
        y1=[]
        m = len(x)
        for i in range(m):
            if(x[i][j]<=medianval):
                x0.append(x[i])
                y0.append(y[i])
            elif(x[i][j]>medianval):
                x1.append(x[i])
                y1.append(y[i])
        x0 = np.array(x0)
        y0 = np.array(y0)
        x1 = np.array(x1)
        y1 = np.array(y1)
        currDepth = currDepth+1
        child0 = growTree(x0,y0,dictionary,nodej,currDepth,maxDepth)
        child1 = growTree(x1,y1,dictionary,nodej,currDepth,maxDepth)
        nodej.medianValue = medianval
        nodej.children.append(child0)
        nodej.children.append(child1)
        return nodej
    else:
        # first xj = 0
        # filter data with xj as 0 
        # node child = growtree()
        # add this child to the nodej
        x0=[]
        y0=[]
        x1=[]
        y1=[]
        m = len(x)
        for i in range(m):
            if(x[i][j]==0):
                x0.append(x[i])
                y0.append(y[i])
            elif(x[i][j]==1):
                x1.append(x[i])
                y1.append(y[i])
        x0 = np.array(x0)
        y0 = np.array(y0)
        x1 = np.array(x1)
        y1 = np.array(y1)
        currDepth =currDepth +1
        child0 = growTree(x0,y0,dictionary,nodej,currDepth,maxDepth)
        child1 = growTree(x1,y1,dictionary,nodej,currDepth,maxDepth)
        nodej.children.append(child0)
        nodej.children.append(child1)
        return nodej
        # then xj =1
        # filter data with xj as 1

def countNodes(node):
    if(node.leaf):
        return 1
    else:
        return 1 + countNodes(node.children[0])+countNodes(node.children[1])

def depthOfTree(root):
    if(root.leaf):
        return 1
    else:
        depth0 =depthOfTree(root.children[0])
        depth1 = depthOfTree(root.children[1])
        if(depth0>depth1):
            return 1 + depth0
        else:
            return 1 + depth1

def predictfunc(x,node):
    if(node.leaf):
        return node.leafValue+1
    else:
        if(node.continuous):
            check = node.medianValue
            attr = node.index
            if(x[attr]<=check):
                return predictfunc(x,node.children[0])
            else:
                return predictfunc(x,node.children[1])

        else:
            attr = node.index
            if(x[attr]==0):
                return predictfunc(x,node.children[0])
            else:
                return predictfunc(x,node.children[1])
    return -1

def accuracy(prediction,y):
    numVal = len(y)
    ansval= 0
    for i in range(numVal):
        if(prediction[i]==y[i][0]):
            ansval = ansval +1
    return (float(ansval)/float(numVal))



def predict(testx,root):
    numTests = len(testx)
    predictionTest= [0]*numTests
    anstest = 0 
    for i in range(numTests):
        predictionTest[i] = predictfunc(testx[i],root)
    return predictionTest



def part1(trP, valP, teP,output):
    ((trainingX,trainingY),dictionary)=fileReading(trP)
    ((testx,testy),useless)=fileReading(teP)
    ((valx,valy),useless)=fileReading(valP)


    currDepth = 0
    maxDepth = 50

    
    root = growTree(trainingX,trainingY,dictionary,None,currDepth,maxDepth)
    
    
    
    predictionTest = predict(testx,root)
    showOut = [0]*(len(predictionTest)+1)
    showOut[0] = 5
    for i in range(len(predictionTest)):
        showOut[i+1]= predictionTest[i]
    np.savetxt(output, showOut, fmt="%d", delimiter="\n")

    # print(" test accuracy")
    # testacc = accuracy(predictionTest,testy)
    # print(testacc)

    # predictionTrain = predict(trainingX,root)
    # print(" train accuracy")
    # trainacc = accuracy(predictionTrain,trainingY)
    # print(trainacc)
    

    # predictionVal = predict(valx,root)
    # print(" validation accuracy")
    # trainval = accuracy(predictionVal,valy)
    # print(trainval)


    # print(depthOfTree(root))= 86071
    # print(countNodes(root))= 31


def bfs(root):
    visited = [root]
    ret = []
    while(True):
        if(len(visited)==0):
            break
        top = visited.pop(0)
        ret.append(top)
        if(top.leaf==False):
            child0 = top.children[0]
            child1 = top.children[1]
            visited.append(child0)
            visited.append(child1)
    return ret

def improveTree(xval,yval,root):
    if(root.leaf):
        all = len(yval)
        if(all>0):
            Yarr = [0]*7
            for i in range(all):
                Yarr[yval[i][0]-1] = Yarr[yval[i][0]-1] + 1
            
            hcur = 0.0

            for i in range(7):
                Yarr[i]= Yarr[i]/all
                if(Yarr[i]!=0):
                    hcur = hcur + Yarr[i]*(math.log((1/Yarr[i]),2))

            root.total = all
            root.hval = hcur

            index = np.argmax(Yarr)
            root.highest = Yarr[index]
    else:
        all = len(yval)
        if(all>0):
            Yarr = [0]*7
            for k in range(all):
                Yarr[yval[k][0]-1] = Yarr[yval[k][0]-1] + 1

            hcur = 0.0

            for k in range(7):
                Yarr[k]= Yarr[k]/all
                if(Yarr[k]!=0):
                    hcur = hcur + Yarr[k]*(math.log((1/Yarr[k]),2))

            root.total = all
            root.hval = hcur

            index = np.argmax(Yarr)
            root.highest = Yarr[index]
            if(root.continuous):
                medianval = root.medianValue
                attr = root.index
                x0=[]
                y0=[]
                x1=[]
                y1=[]
                m = len(xval)
                for i in range(m):
                    if(xval[i][attr]<=medianval):
                        x0.append(xval[i])
                        y0.append(yval[i])
                    elif(xval[i][attr]>medianval):
                        x1.append(xval[i])
                        y1.append(yval[i])
                x0 = np.array(x0)
                y0 = np.array(y0)
                x1 = np.array(x1)
                y1 = np.array(y1)
                improveTree(x0,y0,root.children[0])
                improveTree(x1,y1,root.children[1])
            else:
                attr = root.index
                x0=[]
                y0=[]
                x1=[]
                y1=[]
                m = len(xval)
                for i in range(m):
                    if(xval[i][attr]==0):
                        x0.append(xval[i])
                        y0.append(yval[i])
                    elif(xval[i][attr]==1):
                        x1.append(xval[i])
                        y1.append(yval[i])
                x0 = np.array(x0)
                y0 = np.array(y0)
                x1 = np.array(x1)
                y1 = np.array(y1)
                improveTree(x0,y0,root.children[0])
                improveTree(x1,y1,root.children[1])




def part2(trP,valP,teP,output):
    ((trainingX,trainingY),dictionary)=fileReading(trP)
    ((testx,testy),useless)=fileReading(teP)
    ((valx,valy),useless)=fileReading(valP)

    currDepth = 0
    maxDepth = 50

    root = growTree(trainingX,trainingY,dictionary,None,currDepth,maxDepth)

    improveTree(valx,valy,root)
    allNodes = bfs(root)

    allNodes.reverse()
    pruneNodes = 0

    for node in allNodes:
        if(node.leaf==True):
            continue
        if(node.total >0):
            Hnode = node.hval
            total = node.total

            Hchild0 = node.children[0].hval
            total0 = node.children[0].total
            Hchild1 = node.children[1].hval
            total1 = node.children[1].total
            Hchildfinal = (Hchild0*total0/total) + (Hchild1*total1/total)

            if(Hchildfinal >= Hnode):
                node.leaf = True
                pruneNodes+=1


    # print("Total Nodes -> " + str(countNodes(root)))
    # print("Training Data")
    # predicted = predict(trainingX,root)
    # print(accuracy(predicted,trainingY))

    
    predicted = predict(testx,root)
    showOut = [0]*(len(predicted)+1)
    showOut[0] = 5
    for i in range(len(predicted)):
        showOut[i+1]= predicted[i]
    np.savetxt(output, showOut, fmt="%d", delimiter="\n")

    # print("Testing Data")
    # print(accuracy(predicted,testy))

    # print("Validation Data")
    # predicted = predict(valx,root)
    # print(accuracy(predicted,valy))


    # print(node_count)
    # print(trainArr)
    # print(valArr)
    # print(testArr)
    


    # # # fig = plt.figure()
    # plt.title("Accuracies vs Number of Nodes")
    # plt.plot(node_count, trainArr, label = 'Training')
    # plt.plot(node_count, valArr, label = 'Validation')
    # plt.plot(node_count, testArr, label = 'Testing')
    # plt.xlim(max(node_count)+10,min(node_count)-10)
    # plt.xlabel("Number of Nodes")
    # plt.ylabel('Accuracies')
    # plt.legend()
    # plt.show()
    # # # fig.savefig("Graph_Part2"+'.png')


def part3(trP,valP,teP,output):


    
    ((trainingX,trainingY),dictionary)=fileReading(trP)
    ((testx,testy),useless)=fileReading(teP)
    ((valx,valy),useless)=fileReading(valP)
    
    # code to find the best parameters
    # parameters = {'n_estimators':[50,150,250,350],'max_features':[0.1,0.3,0.5,0.7,0.9],'min_samples_split':[2,4,6,8,10]}

    # rfc = RandomForestClassifier(oob_score=True)
    # clf = GridSearchCV(rfc, parameters)
    # trainingY= trainingY.ravel()
    # clf.fit(trainingX, trainingY)
    # print("The best paramters are: ")
    # # print(clf.best_estimator_)
    # print(clf.best_params_)
    # obscore = clf.oob_score_
    # print("obszore")
    # print(obscore)
    # prediction = rfc.predict(testx)
    # ans =0 
    # for i in range(len(prediction)):
    #     if(prediction[i]==testy[i]):
    #         ans = ans +1
    # print("accuracy")
    # print(ans/len(prediction))

    print("part 3")
    rfc = RandomForestClassifier(n_estimators=450,max_features=0.7,min_samples_split=2,oob_score=True)
    trainingY= trainingY.ravel()
    rfc.fit(trainingX, trainingY)
    obscore = rfc.oob_score_

    print("Obscore")
    print(obscore)

    print("Accuracy On Training Data: ")
    prediction = rfc.predict(trainingX)
    ans =0 
    for i in range(len(prediction)):
        if(prediction[i]==trainingY[i]):
            ans = ans +1
    print(ans/len(prediction))

    print("Accuracy On Validation Data: ")
    prediction = rfc.predict(valx)
    ans =0 
    for i in range(len(prediction)):
        if(prediction[i]==valy[i]):
            ans = ans +1
    print(ans/len(prediction))

    print("Accuracy On Test Data: ")
    prediction = rfc.predict(testx)
    ans =0 
    for i in range(len(prediction)):
        if(prediction[i]==testy[i]):
            ans = ans +1
    print(ans/len(prediction))
    

def part4(trP,valP,teP,output):
    print("part 4")
    ((trainingX,trainingY),dictionary)=fileReading(trP)
    ((testx,testy),useless)=fileReading(teP)
    ((valx,valy),useless)=fileReading(valP)

    optimum_numEstimators = 450
    optimum_max_features = 0.1
    optimum_min_samples_split = 2

    # Vary number of estimators 
    estimatorArray = [100,250,450]
    trainarr = [0.0]*3
    valarr= [0.0]*3
    testarr = [0.0]*3
    for k in range(len(estimatorArray)):
        rfc = RandomForestClassifier(n_estimators=estimatorArray[k],max_features=0.1,min_samples_split=2,oob_score=True)
        rfc.fit(trainingX, trainingY)

        prediction = rfc.predict(trainingX)
        ans =0 
        for i in range(len(prediction)):
            if(prediction[i]==trainingY[i]):
                ans = ans +1
        trainarr[k]=(ans/len(prediction))

        prediction = rfc.predict(valx)
        ans =0 
        for i in range(len(prediction)):
            if(prediction[i]==valy[i]):
                ans = ans +1
        valarr[k] = (ans/len(prediction))

        prediction = rfc.predict(testx)
        ans =0 
        for i in range(len(prediction)):
            if(prediction[i]==testy[i]):
                ans = ans +1
        testarr[k]=(ans/len(prediction))
    print("printing for varying num_estimators.")
    print("the arrays are in order of train, val , test:")
    print(trainarr)
    print(valarr)
    print(testarr)

    # # fig = plt.figure()
    plt.title("Accuracy vs Number of Estimators")
    plt.plot(estimatorArray, valarr, label = 'Validation')
    plt.plot(estimatorArray, testarr, label = 'Testing')
    
    
    plt.xlabel("n_estimators")
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    maxFeatureArray = [0.1,0.3,0.5,0.7,0.9]
    trainarr = [0.0]*1
    valarr= [0.0]*1
    testarr = [0.0]*1
    for k in range(len(maxFeatureArray)):
        rfc = RandomForestClassifier(n_estimators=450,max_features=maxFeatureArray[k],min_samples_split=2,oob_score=True)
        rfc.fit(trainingX, trainingY)

        prediction = rfc.predict(trainingX)
        ans =0 
        for i in range(len(prediction)):
            if(prediction[i]==trainingY[i]):
                ans = ans +1
        trainarr[k]=(ans/len(prediction))

        prediction = rfc.predict(valx)
        ans =0 
        for i in range(len(prediction)):
            if(prediction[i]==valy[i]):
                ans = ans +1
        valarr[k] = (ans/len(prediction))

        prediction = rfc.predict(testx)
        ans =0 
        for i in range(len(prediction)):
            if(prediction[i]==testy[i]):
                ans = ans +1
        testarr[k]=(ans/len(prediction))
    print("printing for varying max_features.")
    print("the arrays are in order of train, val , test:")
    print(trainarr)
    print(valarr)
    print(testarr)

    plt.title("Accuracy vs Max Features")
    plt.plot(maxFeatureArray, valarr, label = 'Validation')
    plt.plot(maxFeatureArray, testarr, label = 'Testing')
    
    
    plt.xlabel("max_features")
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    minSplitArr = [2,4,6]
    trainarr = [0.0]*3
    valarr= [0.0]*3
    testarr = [0.0]*3
    for k in range(len(minSplitArr)):
        rfc = RandomForestClassifier(n_estimators=450,max_features=0.1,min_samples_split=minSplitArr[k],oob_score=True)
        rfc.fit(trainingX, trainingY)

        prediction = rfc.predict(trainingX)
        ans =0 
        for i in range(len(prediction)):
            if(prediction[i]==trainingY[i]):
                ans = ans +1
        trainarr[k]=(ans/len(prediction))

        prediction = rfc.predict(valx)
        ans =0 
        for i in range(len(prediction)):
            if(prediction[i]==valy[i]):
                ans = ans +1
        valarr[k] = (ans/len(prediction))

        prediction = rfc.predict(testx)
        ans =0 
        for i in range(len(prediction)):
            if(prediction[i]==testy[i]):
                ans = ans +1
        testarr[k]=(ans/len(prediction))
    print("printing for varying min_samples_split.")
    print("the arrays are in order of train, val , test:")
    print(trainarr)
    print(valarr)
    print(testarr)

    plt.title("Accuracy vs Min Samples Split")
    plt.plot(minSplitArr, valarr, label = 'Validation')
    plt.plot(minSplitArr, testarr, label = 'Testing')
    
    
    plt.xlabel("min_samples_split")
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()



def main():
    part = sys.argv[1]
    trP = sys.argv[2]
    valP = sys.argv[3]
    teP = sys.argv[4]
    output = sys.argv[5]

    if part=="1":
        part1(trP, valP, teP,output)
    if part =="2":
        part2(trP,valP,teP,output)
    if part =="3":
        part3(trP,valP,teP,output)
    if part == "4":
        part4(trP,valP,teP,output)


if __name__ == "__main__":
    main()