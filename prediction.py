from sklearn import tree # Import the tree module from the sklearn library

from sklearn import neural_network # Import the neural_network module from the sklearn library

from sklearn import svm # Import the svm module from the sklearn library

# Height, weight, shoe size
x = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40]] 

y = ['male', 'male', 'female', 'female', 'female']

clf = tree.DecisionTreeClassifier()
clf1 = neural_network.MLPClassifier()
clf2 = svm.SVC() 

clf = clf.fit(x, y)
clf1 = clf1.fit(x, y)
clf2 = clf2.fit(x, y)


prediction = clf.predict([[160, 60, 40]])
prediction1 = clf1.predict([[180, 75, 42]])
prediction2 = clf2.predict([[160, 40, 35]])

print(prediction)
print(prediction1)
print(prediction2)