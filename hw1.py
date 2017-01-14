from mnist import MNIST
import numpy as np

# load the MNIST data set
nTrain = 20000
nTest = 2000
mndata = MNIST('.')
mndata.load_training()
mndata.load_testing()

# using numpy array form to store the train and test sets
# and transform the data type to double
train_images = np.array(mndata.train_images[:nTrain]).T / 255.
train_labels = np.array(mndata.train_labels[:nTrain]).T
test_images = np.array(mndata.test_images[:nTest]).T / 255.
test_labels = np.array(mndata.test_labels[:nTest]).T


# pre-process the data so that each pixel will have roughly zero mean and unit variance.
def preproc(images):
    m = np.mean(images)
    s = np.std(images)
    images -= m
    images /= (s + 0.1)
    return images


# class of logistic
class Logistic():
    # pick out expected numbers
    @staticmethod
    def select_numbers(images, labels, num1, num2):
        mask = np.any([labels==num1, labels==num2], axis=0)
        images, labels = images[:, mask], labels[mask]
        labels[labels==num1], labels[labels==num2] = 1, 0
        return images, labels

    # randomly shuffle the data
    @staticmethod
    def shuffle(images, labels):
        idx = np.random.permutation(labels.size)
        images, labels = images[:, idx], labels[idx]
        return images, labels

    # insert an extra dimension for the bias term
    @staticmethod
    def insert_bias(images):
        images = np.insert(images, images.shape[0], np.array([1]), axis=0)
        return images

    # initialization
    # fit the model using gradient descent
    def __init__(self, train_images, train_labels, num1=2, num2=3, learning_rate=1., anneal=-1, early_stop=False):
        self.num1, self.num2 = num1, num2
        self.images, self.labels = self.select_numbers(train_images, train_labels, num1, num2)
        self.images, self.labels = self.shuffle(self.images, self.labels)
        self.images = preproc(self.images)
        self.images = np.insert(self.images, self.images.shape[0], 1, axis=0)
        if early_stop:
            hold_out_size = test_images.shape[1] / 10
            self.hold_out = self.images[:, :hold_out_size]
            self.images = self.images[:, hold_out_size:]
        self.weight = self.gradient_descent(self.images, self.labels, learning_rate, anneal, early_stop)

    # gradient descent
    @staticmethod
    def gradient_descent(x, t, learning_rate, anneal, early_stop):
        d, n = x.shape
        w = 1e-3 * np.random.randn(1, d)
        for i in xrange(200):
            scores = np.dot(w, x)
            y = 1 / (1 + np.exp(-scores))
            loss = -np.sum(t * np.log(y) + (1 - t) * np.log(1 - y)) / n
            dw = np.dot(t - y, x.T) / n
            if anneal == -1:
                w += learning_rate * dw
            else:
                if i <= 5:
                    anneal_lr = 1e-1
                else:
                    anneal_lr = learning_rate / (1. + i / anneal)
                w += anneal_lr * dw
            if i % 100 == 0:
                print loss
        return w

    # test on the test set
    def test(self, test_images, test_labels):
        test_images = preproc(test_images)
        test_images, test_labels = self.select_numbers(test_images, test_labels, self.num1, self.num2)
        test_images = self.insert_bias(test_images)
        scores = np.dot(self.weight, test_images)
        y = 1 / (1 + np.exp(-scores)) > 0.5
        error = np.mean(np.abs(y - test_labels))
        return error


logistic = Logistic(train_images, train_labels, learning_rate=1e0, anneal=1, early_stop=False)
print ""
print logistic.test(test_images, test_labels)
