'''
MIT License

Copyright (c) 2012-2015 Michael Nielsen

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software 
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN 
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import loader

# Third-party libraries
from sklearn import svm
from sklearn.externals import joblib

def svm_baseline():
    training_data, validation_data, test_data = loader.read_data_sets()
    # train
    clf = svm.SVC()
    clf.fit(training_data.images, training_data.labels)
    # test
    predictions = [int(a) for a in clf.predict(test_data.images)]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data.labels))
    print ("Baseline classifier using an SVM.")
    print ("%s of %s values correct." % (num_correct, len(test_data.labels)))
    # Save model.
    joblib.dump(clf, 'models/model.pkl') 

if __name__ == "__main__":
    svm_baseline()
    
