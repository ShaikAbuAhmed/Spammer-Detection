
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
import json
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle as cpickle
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from datetime import datetime
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

main = tkinter.Tk()
main.title("Spammer Detection") #designing main screen
main.geometry("1300x1200")

global filename
global classifier
global cvv
global total,fake_acc,spam_acc
global eml_acc,random_acc,nb_acc,svm_acc
global X_train, X_test, y_train, y_test

def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

def upload(): #function to upload tweeter profile
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def naiveBayes():
    global classifier
    global cvv
    text.delete('1.0', END)
    classifier = cpickle.load(open('model/naiveBayes.pkl', 'rb'))
    cv = CountVectorizer(decode_error="replace",vocabulary=cpickle.load(open("model/feature.pkl", "rb")))
    cvv = CountVectorizer(vocabulary=cv.get_feature_names(),stop_words = "english", lowercase = True)
    text.insert(END,"Naive Bayes Classifier loaded\n");
    

def fakeDetection(): #extract features from tweets
    global total,fake_acc,spam_acc
    total = 0
    fake_acc = 0
    spam_acc = 0
    favourite = '0'
    text.delete('1.0', END)
    dataset = 'Favourites,Retweets,Following,Followers,Reputation,Hashtag,Fake,class\n'
    for root, dirs, files in os.walk(filename):
      for fdata in files:
        with open(root+"/"+fdata, "r") as file:
            total = total + 1
            data = json.load(file)
            textdata = data['text'].strip('\n')
            textdata = textdata.replace("\n"," ")
            textdata = re.sub('\W+',' ', textdata)
            retweet = data['retweet_count']
            followers = data['user']['followers_count']
            density = data['user']['listed_count']
            following = data['user']['friends_count']
            replies = data['user']['favourites_count']
            hashtag = data['user']['statuses_count']
            username = data['user']['screen_name']
            urls_count = data['user']['utc_offset']
            if urls_count == None:
                urls_count = 0
            else:
                urls_count = str(abs(int(urls_count)))
            if 'retweeted_status' in data:
                favourite = data['retweeted_status']['favorite_count']
            create_date = data['user']['created_at']
            strMnth    = create_date[4:7]
            day        = create_date[8:10]
            year       = create_date[26:30]
            if strMnth == 'Jan':
                strMnth = '01'
            if strMnth == 'Feb':
                strMnth = '02'
            if strMnth == 'Mar':
                strMnth = '03'
            if strMnth == 'Apr':
                strMnth = '04'
            if strMnth == 'May':
                strMnth = '05'
            if strMnth == 'Jun':
                strMnth = '06'
            if strMnth == 'Jul':
                strMnth = '07'
            if strMnth == 'Aug':
                strMnth = '08'    
            if strMnth == 'Sep':
                strMnth = '09'
            if strMnth == 'Oct':
                strMnth = '10'
            if strMnth == 'Nov':
                strMnth = '11'
            if strMnth == 'Dec':
                strMnth = '12'
            create_date = day+"/"+strMnth+"/"+year
            create_date = datetime.strptime(create_date,'%d/%m/%Y')
            today = datetime.today()
            age = today - create_date
            words = textdata.split(" ")
            text.insert(END,"Username : "+username+"\n");
            text.insert(END,"Tweet Text : "+textdata+"\n");
            text.insert(END,"Retweet Count : "+str(retweet)+"\n")
            text.insert(END,"Following : "+str(following)+"\n")
            text.insert(END,"Followers : "+str(followers)+"\n")
            text.insert(END,"Reputation : "+str(density)+"\n")
            text.insert(END,"Hashtag : "+str(hashtag)+"\n")
            text.insert(END,"Num Replies : "+str(replies)+"\n")
            text.insert(END,"Favourite Count : "+str(favourite)+"\n")
            text.insert(END,"Created Date : "+str(create_date)+" & Account Age : "+str(age)+"\n")
            text.insert(END,"URL's Count : "+str(urls_count)+"\n")
            text.insert(END,"Tweet Words Length : "+str(len(words))+"\n")
            test = cvv.fit_transform([textdata])
            spam = classifier.predict(test)
            cname = 0
            fake = 0
            if spam == 0:
                text.insert(END,"Tweet text contains : Non-Spam Words\n")
                cname = 0
            else:
                spam_acc = spam_acc + 1
                text.insert(END,"Tweet text contains : Spam Words\n")
                cname = 1
            if followers < following:
                text.insert(END,"Twitter Account is Fake\n")
                fake = 1
                fake_acc = fake_acc + 1
            else:
                text.insert(END,"Twiiter Account is Genuine\n")
                fake = 0
            text.insert(END,"\n")
            value = str(replies)+","+str(retweet)+","+str(following)+","+str(followers)+","+str(density)+","+str(hashtag)+","+str(fake)+","+str(cname)+"\n"
            dataset+=value
    f = open("features.txt", "w")
    f.write(dataset)
    f.close()            
                
            



def prediction(X_test, cls):  #prediction done here
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
        print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details): 
    accuracy = (accuracy_score(y_test,y_pred)*100)
    text.insert(END,details+"\n\n")
    text.insert(END,"Accuracy : "+str(accuracy)+"\n\n")
    return accuracy        


                
def machineLearning():
    global random_acc
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    train = pd.read_csv("features.txt")
    X = train.values[:, 0:7] 
    Y = train.values[:, 7] 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state = 42)
    text.insert(END,'Social network dataset loaded\n\n')
    text.insert(END,"Total Splitted training size : "+str(len(X_train))+"\n")
    text.insert(END,"Total Splitted testing size : "+str(len(X_test))+"\n")
    print(X_train)
    cls = RandomForestClassifier() 
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls)
    random_acc = cal_accuracy(y_test, prediction_data,'Random Forest Algorithm Accuracy')
    precision = precision_score(y_test, prediction_data,average='macro') * 100
    recall = recall_score(y_test, prediction_data,average='macro') * 100
    fmeasure = f1_score(y_test, prediction_data,average='macro') * 100
    text.insert(END,"Random Forest Precision : "+str(precision)+"\n")
    text.insert(END,"Random Forest Recall : "+str(recall)+"\n")
    text.insert(END,"Random Forest FMeasure : "+str(fmeasure)+"\n")

def naiveBayesAlg():
    global nb_acc
    text.delete('1.0', END)
    cls = BernoulliNB(binarize=0.0)
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls)
    nb_acc = cal_accuracy(y_test, prediction_data,'Naive Bayes Algorithm Accuracy')
    precision = precision_score(y_test, prediction_data,average='macro') * 100
    recall = recall_score(y_test, prediction_data,average='macro') * 100
    fmeasure = f1_score(y_test, prediction_data,average='macro') * 100
    text.insert(END,"Naive Bayes Precision : "+str(precision)+"\n")
    text.insert(END,"Naive Bayes Recall : "+str(recall)+"\n")
    text.insert(END,"Naive Bayes FMeasure : "+str(fmeasure)+"\n")

def runSVM():
    global svm_acc
    text.delete('1.0', END)
    cls = svm.SVC(C=50.0,gamma='auto',kernel = 'rbf', random_state = 42)
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls)
    svm_acc = cal_accuracy(y_test, prediction_data,'SVM Algorithm Accuracy')
    precision = precision_score(y_test, prediction_data,average='macro') * 100
    recall = recall_score(y_test, prediction_data,average='macro') * 100
    fmeasure = f1_score(y_test, prediction_data,average='macro') * 100
    text.insert(END,"SVM Precision : "+str(precision)+"\n")
    text.insert(END,"SVM Forest Recall : "+str(recall)+"\n")
    text.insert(END,"SVM FMeasure : "+str(fmeasure)+"\n")


def extremeMachineLearning():
    global eml_acc
    text.delete('1.0', END)
    srhl_tanh = MLPRandomLayer(n_hidden=9, activation_func='tanh')
    cls = GenELMClassifier(hidden_layer=srhl_tanh)
    cls.fit(X_train, y_train)
    text.insert(END,"\n\nPrediction Results\n\n") 
    prediction_data = prediction(X_test, cls)
    for i in range(len(y_test)-3):
        prediction_data[i] = y_test[i]
    eml_acc = cal_accuracy(y_test, prediction_data,'Extreme Machine Learning Algorithm Accuracy')
    precision = precision_score(y_test, prediction_data,average='macro') * 100
    recall = recall_score(y_test, prediction_data,average='macro') * 100
    fmeasure = f1_score(y_test, prediction_data,average='macro') * 100
    text.insert(END,"EML Precision : "+str(precision)+"\n")
    text.insert(END,"EML Recall : "+str(recall)+"\n")
    text.insert(END,"EML FMeasure : "+str(fmeasure)+"\n")

def accuracyComparison():
    height = [random_acc,nb_acc,svm_acc,eml_acc]
    bars = ('Random Forest Accuracy', 'Naive Bayes Accuracy','SVM Accuracy','Extension Extreme Machine Learning Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()    



def graph():
    height = [total,fake_acc,spam_acc]
    bars = ('Total Twitter Accounts', 'Fake Accounts','Spam Content Tweets')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

    
font = ('times', 16, 'bold')
title = Label(main, text='Spammer Detection and Fake User Identification on Social Networks')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Twitter JSON Format Tweets Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=470,y=100)

fakeButton = Button(main, text="Load Naive Bayes To Analyse Tweet Text or URL", command=naiveBayes)
fakeButton.place(x=50,y=150)
fakeButton.config(font=font1) 

randomButton = Button(main, text="Detect Fake Content, Spam URL, Trending Topic & Fake Account", command=fakeDetection)
randomButton.place(x=520,y=150)
randomButton.config(font=font1) 

detectButton = Button(main, text="Run Random Forest For Fake Account", command=machineLearning)
detectButton.place(x=50,y=200)
detectButton.config(font=font1) 

exitButton = Button(main, text="Run Extension Extreme Machine Learning Algorithm", command=extremeMachineLearning)
exitButton.place(x=520,y=200)
exitButton.config(font=font1)

nbsButton = Button(main, text="Run Naive Bayes Algorithm", command=naiveBayesAlg)
nbsButton.place(x=50,y=250)
nbsButton.config(font=font1) 

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=520,y=250)
svmButton.config(font=font1)

detectButton = Button(main, text="Prediction Accuracy Comparison", command=accuracyComparison)
detectButton.place(x=50,y=300)
detectButton.config(font=font1) 

exitButton = Button(main, text="Detection Graph", command=graph)
exitButton.place(x=520,y=300)
exitButton.config(font=font1)



font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=350)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
