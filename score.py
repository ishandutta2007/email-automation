import json, os, pickle
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

vocab =  pickle.load(open('TfidfVectorizerModel.pkl', 'rb'))
model = joblib.load(open('MultinomialNBModel.pkl','rb'))

transformer = TfidfTransformer()
trainedVectorizer = CountVectorizer(decode_error='replace',vocabulary=vocab)

def run(raw_data):
    y_hat = dict()
    vectorString  = trainedVectorizer.fit_transform([str(raw_data)])
    transformedString = transformer.fit_transform(vectorString)
    y_hat['prediction']  = model.predict(transformedString).astype(dtype=float)[0]
    y_hat['probability'] = model.predict_proba(transformedString).astype(dtype=float).tolist()
    return(json.dumps(y_hat))