# import libraries
import json, os, pickle
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# define init function
def init(filepath):
    # define global variable
    global vocab, model, trainedVectorizer, transformer, path

    # find script location
    path = filepath

    # load machine learning model and meta data
    vocab =  pickle.load(open(path + "TfidfVectorizerModel.pkl", "rb"))
    model = joblib.load(open(path + "MultinomialNBModel.pkl","rb"))

    # init stages using meta data
    trainedVectorizer = CountVectorizer(decode_error="replace",vocabulary=vocab)
    transformer = TfidfTransformer()

# define run function to execute the ml model
def run(raw_data):

    # define y_hat (result) dictionary
    y_hat = dict()

    # transform feature to feature vector
    featureVector = trainedVectorizer.fit_transform([raw_data])
    featureVector_fit = transformer.fit_transform(featureVector)

    # save result into y_hat
    y_hat["prediction"]  = model.predict(featureVector_fit).astype(dtype=float)[0]
    y_hat["probability"] = max(max(model.predict_proba(featureVector_fit).astype(dtype=float).tolist()))

    # return JSON
    return(json.dumps(y_hat))


if __name__ == "__main__":
    test = "This is to check machine learning model"
    filepath = json.load(open('config.uipath'))['filepath']
    init(filepath)
    result = run(test)
    print("Data: {}\nResult: {}".format(test,result))
