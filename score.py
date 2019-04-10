import json, os, pickle
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

vocab =  pickle.load(open("D:\Demo\Email-Automation-master\TfidfVectorizerModel.pkl", 'rb'))
model = joblib.load(open("D:\Demo\Email-Automation-master\MultinomialNBModel.pkl", 'rb'))

transformer = TfidfTransformer()
trainedVectorizer = CountVectorizer(decode_error='replace',vocabulary=vocab)

def run(raw_data):
    vectorString  = trainedVectorizer.fit_transform([str(raw_data)])
    transformedString = transformer.fit_transform(vectorString)
    y_hat = model.predict(transformedString)
    return(str(y_hat))
