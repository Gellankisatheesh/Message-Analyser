from django.shortcuts import render
from django.http import HttpResponse
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import joblib

def wordopt(text):
    return text.lower()

sms_spam_df = pd.read_csv('sim.csv', quoting=csv.QUOTE_NONE, sep='\t', names=['label', 'message'])
sms_spam_df['class'] = sms_spam_df['label'].map({'spam': 0, 'ham': 1})
sms_spam_df = sms_spam_df.rename(columns={'message': 'text'})

x = sms_spam_df["text"]
y = sms_spam_df["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=100)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
LR.score(xv_test, y_test)

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
DT.score(xv_test, y_test)


GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)
pred_gbc = GBC.predict(xv_test)
GBC.score(xv_test, y_test)

RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
pred_rfc = RFC.predict(xv_test)
RFC.score(xv_test, y_test)

def output_label(n):
    if n == 0:
        return "Fake Message"
    elif n == 1:
        return "Not A Fake Message"

def manual_testing(message):
    testing_news = {"text": [message]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nLR Prediction: {}\nDT Prediction: {}\nGBC Prediction: {}\nRFC Prediction: {}".format(
        output_label(pred_LR[0]),
        output_label(pred_DT[0]),
        output_label(pred_GBC[0]),
        output_label(pred_RFC[0])
    ))
# Save the trained model and vectorization object
joblib.dump(LR, 'logistic_regression_model.pkl')
joblib.dump(DT, 'logistic_regression_model.pkl')
joblib.dump(GBC, 'logistic_regression_model.pkl')
joblib.dump(RFC, 'logistic_regression_model.pkl')
joblib.dump(vectorization, 'vectorization.pkl')

def home(request):
    if request.method == 'POST':
        message = request.POST.get('message', '')
        model = joblib.load('logistic_regression_model.pkl')
        vectorization = joblib.load('vectorization.pkl')
        testing_news = {"text": [message]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test["text"] = new_def_test["text"].apply(wordopt)
        new_x_test = new_def_test["text"]
        new_xv_test = vectorization.transform(new_x_test)
        pred_LR = model.predict(new_xv_test)
        pred_DT = DT.predict(new_xv_test)
        pred_GBC = GBC.predict(new_xv_test)
        pred_RFC = RFC.predict(new_xv_test)
        prediction1 = output_label(pred_LR[0])
        prediction2 = output_label(pred_DT[0])
        prediction3 = output_label(pred_GBC[0])
        prediction4 = output_label(pred_RFC[0])
        return render(request, 'app/home.html', {'message': message, 'prediction1': prediction1, 'prediction2': prediction2, 'prediction3': prediction3, 'prediction4': prediction4})
    else:
        return render(request, 'app/home.html')
