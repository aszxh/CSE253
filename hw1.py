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

# randomly shuffle the data
def shuffle(images, labels):
    idx = np.random.permutation(labels.size)
    images, labels = images[:, idx], labels[idx]
    return images, labels

# insert an extra dimension for the bias term
def insert_bias(images):
    images = np.insert(images, images.shape[0], np.array([1]), axis=0)
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

    # initialization
    # fit the model using gradient descent
    def __init__(self, train_images, train_labels, num1=2, num2=3, learning_rate=1., anneal=-1, early_stop=False):
        self.num1, self.num2 = num1, num2
        self.images, self.labels = self.select_numbers(train_images, train_labels, num1, num2)
        self.images, self.labels = shuffle(self.images, self.labels)
        self.images = preproc(self.images)
        self.images = insert_bias(self.images)
        if early_stop:
            hold_out_size = self.images.shape[1] / 10
            self.hold_out_images, self.hold_out_labels = self.images[:, :hold_out_size], self.labels[:hold_out_size]
            self.images, self.labels = self.images[:, hold_out_size:], self.labels[hold_out_size:]
        self.weight = self.gradient_descent(learning_rate, anneal, early_stop)

    @staticmethod
    def eval_loss_and_error(labels, y):
        loss = -np.sum(labels * np.log(y) + (1 - labels) * np.log(1 - y)) / labels.size
        error = np.mean(np.abs((y > 0.5) - labels))
        return loss, error

    @staticmethod
    def eval_gradient(images, labels, y):
        return np.dot(labels - y, images.T) / labels.size

    # gradient descent
    def gradient_descent(self, learning_rate, anneal, early_stop):
        d, n = self.images.shape
        w = 1e-3 * np.random.randn(1, d)
        up_epoch = pre_error = 0
        for i in xrange(2000):
            y = 1 / (1 + np.exp(-np.dot(w, self.images)))
            train_loss, train_error = Logistic.eval_loss_and_error(self.labels, y)
            if early_stop:
                yh = 1 / (1 + np.exp(-np.dot(w, self.hold_out_images)))
                hold_out_loss, hold_out_error = Logistic.eval_loss_and_error(self.hold_out_labels, yh)
                if i > 0:
                    up_epoch = up_epoch + 1 if hold_out_error >= pre_error else 0
                    if up_epoch == 3:
                        break
                pre_error = hold_out_error
            dw = Logistic.eval_gradient(self.images, self.labels, y)
            if anneal == -1:
                w += learning_rate * dw
            else:
                if i <= 5:
                    anneal_lr = 1e-1
                else:
                    anneal_lr = learning_rate / (1. + i / anneal)
                w += anneal_lr * dw
            if i % 1 == 0:
                print train_loss, train_error, hold_out_loss, hold_out_error if early_stop else train_loss
        return w

    # test on the test set
    def test(self, test_images, test_labels):
        test_images = preproc(test_images)
        test_images, test_labels = self.select_numbers(test_images, test_labels, self.num1, self.num2)
        test_images = insert_bias(test_images)
        scores = np.dot(self.weight, test_images)
        y = 1 / (1 + np.exp(-scores))
        loss = -np.sum(test_labels * np.log(y) + (1 - test_labels) * np.log(1 - y)) / test_labels.size
        error = np.mean(np.abs((y > 0.5) - test_labels))
        return error, loss


logistic = Logistic(train_images, train_labels, learning_rate=1e0, anneal=1, early_stop=True)
print ""
print logistic.test(test_images, test_labels)
