from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt


def load_images():
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
    return train_images, train_labels, test_images, test_labels


# pre-process the data so that each pixel will have roughly zero mean and unit variance.
def whitening(images):
    m = np.mean(images, axis=1)
    s = np.std(images, axis=1)
    images = images.T - m
    images /= (s + 0.1)
    images = images.T
    return images


# randomly shuffle the data
def shuffle(X, t):
    idx = np.random.permutation(t.size)
    X, t = X[:, idx], t[idx]
    return X, t


# insert an extra dimension for the bias term
def insert_bias(X):
    X = np.insert(X, X.shape[0], 1, axis=0)
    return X


# gradient descent
def gradient_descent(method, W, learning_rate, anneal, early_stop, mini_batch):
    d, n = method.X.shape
    batch_size = n / mini_batch
    weights = []
    for i in xrange(2000):
        start, end = 0, batch_size
        for j in xrange(mini_batch):
            X_batch, t_batch = method.X[:, start:end], method.t[start:end]
            train_loss, train_error = method.eval_loss_and_error(method.X, method.t, W)
            test_loss, test_error = method.test(test_images, test_labels, W)
            dW = method.eval_gradient(X_batch, t_batch, W)
            learning_rate = learning_rate if anneal == -1 else learning_rate / (1. + i / anneal) if i * mini_batch + j > 5 else 1e-1
            if early_stop:
                holdout_loss, holdout_error = method.eval_loss_and_error(method.X_holdout, method.t_holdout, W)
                print train_loss, train_error, holdout_loss, holdout_error, test_loss, test_error
                method.losses = np.hstack([method.losses, [[train_loss], [holdout_loss], [test_loss]]])
                method.errors = np.hstack([method.errors, [[train_error], [holdout_error], [test_error]]])

                if i != 0 and method.errors[1, -1] >= method.errors[1, -2]:
                    up_epoch += 1
                    if up_epoch == early_stop:
                        idx = method.errors[1, :].argmin()
                        return weights[idx]
                else:
                    up_epoch = 0
            else:
                method.losses = np.hstack([method.losses, [[train_loss], [test_loss]]])
                method.errors = np.hstack([method.errors, [[train_error], [test_error]]])
                print train_loss, train_error, test_loss, test_error

            weights.append(W)
            W += learning_rate * dW
            start, end = start + batch_size, end + batch_size if j != mini_batch - 1 else n
    return W


def plot_losses(method):
    fig, ax = plt.subplots()
    ax.plot(range(method.losses[0].size), method.losses[0], label="train loss")
    ax.plot(range(method.losses[1].size), method.losses[1], label="hold out loss")
    ax.plot(range(method.losses[2].size), method.losses[2], label="test loss")
    ax.legend()


def plot_errors(method):
    fig, ax = plt.subplots()
    ax.plot(range(method.errors[0].size), 1 - method.errors[0], label="train error")
    ax.plot(range(method.errors[1].size), 1 - method.errors[1], label="hold out error")
    ax.plot(range(method.errors[2].size), 1 - method.errors[2], label="test error")
    ax.legend()


def nesterov_momentum(method, W, learning_rate, mu, anneal, early_stop):
    X, t = method.X, method.t
    v = 0
    weights = []
    for i in xrange(2000):
        train_loss, train_error = method.eval_loss_and_error(X, t, W)
        test_loss, test_error = method.test(test_images, test_labels, W)
        dW = method.eval_gradient(X, t, W)
        learning_rate = learning_rate if anneal == -1 else learning_rate / (1. + i / anneal) if i > 5 else 1e-1
        weights.append(W)
        if early_stop:
            holdout_loss, holdout_error = method.eval_loss_and_error(method.X_holdout, method.t_holdout, W)
            method.losses = np.hstack([method.losses, [[train_loss], [holdout_loss], [test_loss]]])
            method.errors = np.hstack([method.errors, [[train_error], [holdout_error], [test_error]]])
            print train_loss, train_error, holdout_loss, holdout_error, test_loss, test_error

            if i != 0 and method.errors[1, -1] >= method.errors[1, -2]:
                up_epoch += 1
                if up_epoch == early_stop:
                    idx = method.errors[1, :].argmin()
                    return weights[idx]
            else:
                up_epoch = 0
        else:
            method.losses = np.hstack([method.losses, [[train_loss], [test_loss]]])
            method.errors = np.hstack([method.errors, [[train_error], [test_error]]])
            print train_loss, train_error, test_loss, test_error

        v_prev = v
        v = mu * v - learning_rate * dW
        W -= -mu * v_prev + (1 + mu) * v
    return W


# class of logistic
class Logistic:
    # pick out expected numbers
    @staticmethod
    def select_numbers(images, labels, num1, num2):
        mask = np.any([labels==num1, labels==num2], axis=0)
        images, labels = images[:, mask], labels[mask]
        labels[labels==num1], labels[labels==num2] = 1, 0
        return images, labels

    # initialization
    # fit the model using gradient descent
    def __init__(self, train_images, train_labels, num1=2, num2=3, method="GD", learning_rate=1.,
                 anneal=-1., early_stop=0, mini_batch=1, reg_type="", reg_weight=0., momentum=0.9):
        self.num1, self.num2 = num1, num2
        self.X, self.t = self.select_numbers(train_images, train_labels, num1, num2)
        self.X, self.t = shuffle(self.X, self.t)
        self.X = insert_bias(whitening(self.X))
        if early_stop:
            holdout_size = self.X.shape[1] / 10
            self.X_holdout, self.t_holdout = self.X[:, :holdout_size], self.t[:holdout_size]
            self.X, self.t = self.X[:, holdout_size:], self.t[holdout_size:]
        self.reg_type = reg_type
        self.reg_weight = reg_weight
        self.losses = np.array([[], [], []]) if early_stop else np.array([[], []])
        self.errors = np.array([[], [], []]) if early_stop else np.array([[], []])
        w0 = 1e-3 * np.random.randn(1, self.X.shape[0])
        if method == "GD":
            self.weight = gradient_descent(self, w0, learning_rate, anneal, early_stop, mini_batch)
        elif method == "NAG":
            self.weight = nesterov_momentum(self, w0, learning_rate, momentum, anneal, early_stop)

    def eval_loss_and_error(self, X, t, W):
        y = 1 / (1 + np.exp(-np.dot(W, X)))
        loss = -np.sum(t * np.log(y) + (1 - t) * np.log(1 - y)) / t.size
        if self.reg_type == "L2":
            loss += np.sum(W * W) * self.reg_weight
        elif self.reg_type == "L1":
            loss += np.sum(np.abs(W)) * self.reg_weight
        error = np.mean(np.abs((y > 0.5) - t))
        return loss, error

    def eval_gradient(self, X, t, W):
        y = 1 / (1 + np.exp(-np.dot(W, X)))
        gradient = np.dot(t - y, X.T) / t.size
        if self.reg_type == "L2":
            gradient += self.reg_weight * W * 2
        elif self.reg_type == "L1":
            reg_grad = W
            reg_grad[reg_grad>0] = 1
            reg_grad[reg_grad<0] = -1
            gradient += self.reg_weight * reg_grad
        return gradient

    def plot_weight(self):
        weight_graph = self.weight[:, :-1].reshape(28, 28)
        plt.imshow(weight_graph, cmap="gray")
        plt.show()

    # test on the test set
    def test(self, X_test, t_test, weight=0):
        X_test, t_test = self.select_numbers(test_images, test_labels, self.num1, self.num2)
        X_test = insert_bias(whitening(X_test))
        if type(weight) is int:
            return self.eval_loss_and_error(X_test, t_test, self.weight)
        else:
            return self.eval_loss_and_error(X_test, t_test, weight)


class Softmax():
    def __init__(self, train_images, train_labels, nClasses=10, learning_rate=1., anneal=-1., early_stop=5,
                 mini_batch=1, reg_type="", reg_weight=0., method="GD", momentum=0.9):
        self.X, self.t = shuffle(train_images, train_labels)
        self.X = whitening(self.X)
        self.X = insert_bias(self.X)
        if early_stop:
            holdout_size = self.X.shape[1] / 10
            self.X_holdout, self.t_holdout = self.X[:, :holdout_size], self.t[:holdout_size]
            self.X, self.t = self.X[:, holdout_size:], self.t[holdout_size:]
        self.reg_type = reg_type
        self.reg_weight = reg_weight
        self.losses = np.array([[], [], []]) if early_stop else np.array([[], []])
        self.errors = np.array([[], [], []]) if early_stop else np.array([[], []])
        w0 = 1e-3 * np.random.randn(nClasses, self.X.shape[0])
        if method == "GD":
            self.weight = gradient_descent(self, w0, learning_rate, anneal, early_stop, mini_batch)
        elif method == "NAG":
            self.weight = nesterov_momentum(self, w0, learning_rate, momentum, anneal, early_stop)

    def eval_loss_and_error(self, X, t, W):
        n = t.size
        scores = np.exp(np.dot(W, X))
        probs = scores / np.sum(scores, axis=0, keepdims=True)
        loss = -np.sum(np.log(probs[t, range(n)])) / n
        if self.reg_type == "L2":
            loss += np.sum(W * W) * self.reg_weight
        elif self.reg_type == "L1":
            loss += np.sum(np.abs(W)) * self.reg_weight
        predict = probs.argmax(axis=0)
        error = np.mean(predict!=t)
        return loss, error

    def eval_gradient(self, X, t, W):
        n = t.size
        scores = np.exp(np.dot(W, X))
        probs = scores / np.sum(scores, axis=0, keepdims=True)
        probs[t, range(n)] -= 1
        gradient = -np.dot(probs, X.T) / n
        if self.reg_type == "L2":
            gradient += 2 * W * self.reg_weight
        elif self.reg_type == "L1":
            reg_grad = W
            reg_grad[reg_grad>0] = 1
            reg_grad[reg_grad<0] = -1
            gradient += self.reg_weight * reg_grad
        return gradient

    def test(self, X_test, t_test, weight=0):
        X_test = insert_bias(whitening(X_test))
        if type(weight) is int:
            return self.eval_loss_and_error(X_test, t_test, self.weight)
        else:
            return self.eval_loss_and_error(X_test, t_test, weight)

if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_images()
    logistic = Logistic(train_images, train_labels, learning_rate=1e0, anneal=1e1, early_stop=3, mini_batch=100)
    print logistic.test(test_images, test_labels)
    plot_losses(logistic)
    plot_errors(logistic)
    plt.show()
    #logistic.plot_weight()
    #softmax = Softmax(train_images, train_labels, learning_rate=1, anneal=1e1, reg_type="L2", reg_weight=1e-4, mini_batch=10)
    #softmax = Softmax(train_images, train_labels, method="NAG", anneal=1, early_stop=5, reg_type="L2", reg_weight=1e-1)
    #print softmax.test(test_images, test_labels)
