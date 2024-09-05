import pickle
# load
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_iris(features):
    
    prediction = model.predict([features])
    return prediction[0]
