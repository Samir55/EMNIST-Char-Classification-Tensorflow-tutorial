import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder
from scipy.io import loadmat
from scipy.optimize import minimize


def load_data(filename):
    """ Loads the MNIST data from the given Matlab file name. """
    try:
        return loadmat(filename)
    except TypeError:
        print("Not a valid filename argument: " + filename)


def sigmoid(x, derive=False):
    """ Calculates the sigmoid function of the given argument. This argument can be a number or a numpy array.
    Also, if you pass a second boolean argument, you will get the derived version of the sigmoid function. """

    if derive:
        return x * (1 - x)

    return 1/(1 + np.exp((-x)))


def sigmoid_gradient(z):
    """ Computes the gradient for the sigmoid of Z - the matrix given as argument. """
    sig_z = sigmoid(z)
    return np.multiply(sig_z, (1 - sig_z))


def forwardprop(X, theta1, theta2):
    """ The implementation for the forward propagation algorithm with one input layer, one hidden layer and
    one output layer. """

    m = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)  # add the ones for the bias unit to the original X matrix and
    #  we get the A1 matrix of input values
    z2 = a1 * theta1.T  # multiply the input values with the initial theta params transposed - this value will be
    # passed through the sigmoid function and we'll get the activation values for the hidden layer
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)  # run z2 through the sigmoid function and add the
    # ones for the bias unit of the hidden layer
    z3 = a2 * theta2.T  # multiply the hidden layer values with the theta2 transpose
    h = sigmoid(z3)  # these are the forward propagation predictions

    return a1, z2, a2, z3, h


def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate, regularize = True):
    """ This is the back propagation algorithm for the particular MLP we are trying to implement here.

    Quote from the source article coming below (http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-5/):
    The first half of the function is calculating the error by running the data plus current parameters through the
    "network" (the forward-propagate function) and comparing the output to the true labels. The total error across the
    whole data set is represented as J.
    The rest of the function is essentially answering the question "how can I adjust my parameters to reduce the error
    the next time I run through the network"? It does this by computing the contributions at each layer to the total
    error and adjusting appropriately by coming up with a "gradient" matrix (or, how much to change each parameter and
    in what direction).

    """
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer - this is similar to the Octave/Matlab code
    # but I'll have to experiment a bit and see how these numpy functions work
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run forward prop
    # a1.shape = (5000, 401)
    # a2.shape = (5000, 26)
    # z2.shape = (5000, 25)
    # h.shape = (5000, 10)
    # y.shape = (5000, 10)
    a1, z2, a2, z3, h = forwardprop(X, theta1, theta2)

    # initializations
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    J = cost(params, input_size, hidden_size, num_labels, X, y, learning_rate, regularize)

    d3 = h - y  # (5000, 10)

    z2 = np.insert(z2, 0, values=np.ones(1), axis=1)  # (5000, 26)

    d2 = np.multiply((theta2.T * d3.T).T, sigmoid_gradient(z2))  # (5000, 26)

    delta1 += (d2[:, 1:]).T * a1
    delta2 += d3.T * a2

    delta1 = delta1 / m
    delta2 = delta2 / m

    # add regularization term if needed
    if regularize:
        delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
        delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad


def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate, regularize=True):
    """ This implements the cost function for determining the error of the neural net predictions. It is needed for
    computing the accuracy of the predictions when testing the network as well as for the backpropagation algorithm,
    to learn the network params. We need the initial set of network params as well as the input layer size, the hidden
    layer size the number of labels and the learning rate lambda.
    I've added the regularize parameter which is by default true. This will add regularization tot he cost function."""

    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)  # make matrices out of the input value and the outputs? - Have to see what this does exactly

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    h = forwardprop(X, theta1, theta2)[4]

    J = (np.multiply(-y, np.log(h)) - np.multiply((1 - y), np.log(1 - h))).sum() / m

    if regularize:
        J += (float(learning_rate) /
              (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    return J


def run_net():
    """ Runs the neural net training sequence and displays the results. The neural net implemented here is based on the
    exercises for the Coursera Machine Learning course held by professor Andrew Ng. The net has 3 layers,
    an input layer, a hidden layer and an output layer and it is supposed to use MNIST data to train itself for
    classifying hand written digits. The data is loaded from the ex3data1.mat Matlab format file available from the
    original course material exercise. The data is loaded from 20x20 greyscale images and we are using 5000 images as
    input data. The input layer will be represented as a 5000x400 matrix (excluding the bias unit), as the 20x20 pixel
    matrices are unrolled in a vector. The hidden layer will have 25 units with one bias unit and the output layer will
    have 10 units corresponding to the one-hot encoding for the class labels (e.g if unit 5 has a result of 1, then it
    means the net predicted that the corresponding input contained the digit 5 in the image).
    I do have this implemented as an Octave/Matlab script as required in the course, but I wanted to try it in Python
    also so I found an implementation on http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-5/ and
    I based my work on that.
    After running the net I got an accuracy of 99.48% and the vectorized approach is much faster than the for loops
    one.
    """

    input_size = 400
    hidden_size = 25
    num_labels = 10
    learning_rate = 1

    data = load_data('data/ex3data1.mat')
    X = data['X']  # loads the X matrix of initial data input for the neural net.
    y = data['y']  # loads the y vector of labels, with corresponding classes for each of the labeled examples from X.

    print(X.shape, y.shape)

    encoder = OneHotEncoder(sparse=False)
    y_encoded = encoder.fit_transform(y)

    print(y_encoded.shape)

    # randomly initialize a parameter array of the size of the full network's parameters
    params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

    print("Running the minimization algorithm for the neural net backpropagation algorithm...")
    start_time = time.time()

    fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_encoded, learning_rate),
                    method='TNC', jac=True, options={'maxiter': 250})

    print("The minimization result: ", fmin)

    X = np.matrix(X)
    theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    a1, z2, a2, z3, h = forwardprop(X, theta1, theta2)
    y_pred = np.array(np.argmax(h, axis=1) + 1)

    print("The network predicts: ", y_pred)

    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))

    print('accuracy = {0}%'.format(accuracy * 100))
    end_time = time.time() - start_time
    print("optimization took: {0} seconds".format(end_time))


if __name__ == "__main__":
    np.random.seed(3)
    run_net()
