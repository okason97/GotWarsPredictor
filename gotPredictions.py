from sklearn import tree
from sklearn import preprocessing

#this is to give each house a number to easily classify them
le = preprocessing.LabelEncoder()
le.fit(['Lannister','Stark','noone', 'Tully', 'Baratheon', 'Darry', 'Greyjoy', 'Night\'s Watch',
       'Brave Companions', 'Mallister', 'Bolton', 'Karstark', 'Mormont', 'Thenns', 'Free folk', 'Blackwood', 
       'Darry', 'Brotherhood without Banners', 'Giants', 'Frey', 'Tyrell', 'Glover', 'Bracken'])

def extract_data(filename):
	
    #labels and features vectors
    labels = []
    fvecs = []

    #first line has atribute names
    iterfile = iter(file(filename))
    next(iterfile)
  
    #extract from file names for fvects and result of war for labels
    for line in iterfile:
        row = line.split(',')
        labels.append(str(row[13]))
        fvecs.append(le.transform([str(x) for x in row[5:13]]))

    return labels, fvecs

#extract battle data
#0 = result
#1 = [[atk1,atk2,atk3,atk4][def1,def2,def3,def4]]
data = extract_data('./battles.csv')

#clf = clasifier
clf = tree.DecisionTreeClassifier()

#fill decision tree
clf = clf.fit(data[1],data[0])

#predict using chosen data, in this case i'm predicting the outcome of a war between Lannister and Stark fammilies
prediction_data = le.transform(['Lannister', 'noone', 'noone', 'noone', 'Mormont', 'Stark', 'noone', 'noone'])
prediction = clf.predict(prediction_data)

print (prediction)
