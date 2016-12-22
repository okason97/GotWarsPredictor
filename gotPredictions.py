from sklearn import tree
from sklearn import preprocessing
from numpy import ndarray

le = preprocessing.LabelEncoder()
le.fit(['Lannister','Stark','noone', 'Tully', 'Baratheon', 'Darry', 'Greyjoy', 'Night\'s Watch',
       'Brave Companions', 'Mallister', 'Bolton', 'Karstark', 'Mormont', 'Thenns', 'Free folk', 'Blackwood', 
       'Darry', 'Brotherhood without Banners', 'Giants', 'Frey', 'Tyrell', 'Glover', 'Bracken'])

def extract_data(filename):
	
    #arreglos con las etiquetas y los vectores con atributos
    labels = []
    fvecs = []

    #salteo los nombres de los atributos
    iterfile = iter(file(filename))
    next(iterfile)
  
    #extraigo del archivo las etiquetas y vectores que necesito
    #le.fit_transform hashea los nombres de las familias
    for line in iterfile:
        row = line.split(',')
        labels.append(str(row[13]))
        fvecs.append(le.transform([str(x) for x in row[5:13]]))

    return labels, fvecs

#extraigo la data de batallas de got
#0 = resultado
#1 = [[atk1,atk2,atk3,atk4][def1,def2,def3,def4]]
data = extract_data('./battles.csv')

#clf = clasifier
clf = tree.DecisionTreeClassifier()

clf = clf.fit(data[1],data[0])

prediction_data = le.transform(['Lannister', 'noone', 'noone', 'noone', 'Mormont', 'Stark', 'noone', 'noone'])
prediction = clf.predict(prediction_data)

print (prediction)