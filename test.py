import os
import pickle

scores = {} # scores is an empty dict already

if os.path.getsize('/home/andrey/Desktop/ml3/wine/data.pkl') > 0:      
    with open('/home/andrey/Desktop/ml3/wine/data.pkl', "rb") as f:
        unpickler = pickle.Unpickler(f)
        # if file is not empty scores will be equal
        # to the value unpickled
        scores = unpickler.load()
