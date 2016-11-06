
import numpy as np


class RBMModel:
    def __init__(self):
        pass

    '''
        :param visible_num number of visible layer
        :param hidden_num number of hidden layer
        :param learning_rate learning rate
        :param min_batch each minimum samples to train
    '''
    def __init__(self, visible_num, hidden_num, learning_rate=0.1, min_batch=100):
        self.visible_num = visible_num
        self.hidden_num = hidden_num
        self.learning_rate = learning_rate
        self.min_batch = min_batch
        # Initialize a weight matrix, of dimensions (num_visible * num_hidden), using
        # a Gaussian distribution with mean 0 and standard deviation 0.1.
        self.weights = 0.1 * np.random.randn(self.visible_num, self.hidden_num)
        # Insert weights for the bias units into the first row and first column.
        self.weights = np.insert(self.weights, 0, 0, axis=0)
        self.weights = np.insert(self.weights, 0, 0, axis=1)

    '''
        :param x sample data
        :return activation state
    '''
    def activation_logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    '''
        :param train_set A matrix where each row is a training example consisting
                        of the states of visible units.
        :param iters number of iterations
        :param eps threshold of parameters
    '''
    def train(self, train_set, iters=1000, eps=1.0e-4):
        for step in xrange(iters):
            error = 0.0
            for i in xrange(0, train_set.shape[0], self.min_batch):
                # select minimum batch of training sample
                num_examples = min(self.min_batch, train_set.shape[0]-i)
                data = train_set[i: i+num_examples, :]
                # Insert bias units of 1 into the first column.
                data = np.insert(data, 0, 1, axis=1)

                # positive CD phase
                # calculate hidden layer states with P(h=1|v)*v
                # clamp to the data and sample from the hidden units.
                # first we calculate P(h=1|v)
                pos_hidden_activations = np.dot(data, self.weights)
                pos_hidden_probs = self.activation_logistic(pos_hidden_activations)
                pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.hidden_num+1)
                # second we calculate P(h=1|v)*v

                # alternative following pos_associations, one is binarization, the other is no binarization
                pos_associations = np.dot(data.T, pos_hidden_states)  # binarization
                # pos_associations = np.dot(data.T, pos_hidden_probs)  # no binarization

                # negative CD phase
                # calculate ∑(P(v))*P(h=1|v)*v
                # Reconstruct the visible units and sample again from the hidden units.
                neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
                neg_visible_probs = self.activation_logistic(neg_visible_activations)
                neg_visible_probs[: 0] = 1
                neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
                neg_hidden_probs = self.activation_logistic(neg_hidden_activations)
                neg_hidden_states = neg_hidden_probs > np.random.rand(num_examples, self.hidden_num+1)
                neg_associations = np.dot(neg_visible_probs.T, neg_hidden_states)
                # neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

                # update weights
                self.weights += self.learning_rate * (pos_associations - neg_associations) / num_examples
                # error function
                error += np.sum((data - neg_visible_probs)**2)
            if error < eps:
                break
            print 'iteration %d, error is %f' % (step, error)

    '''
        :param visible_data A matrix where each row consists of the states of the visible units.
        :return A matrix where each row consists of the hidden units activated from the visible
                units in the data matrix passed in.
    '''
    def get_hidden_layer(self, visible_data):
        num_examples = visible_data.shape[0]
        hidden_states = np.ones((num_examples, self.hidden_num + 1))
        visible_data = np.insert(visible_data, 0, 1, axis=1)
        hidden_activations = np.dot(visible_data, self.weights)
        hidden_probs = self.activation_logistic(hidden_activations)
        hidden_states[:, :] = hidden_probs > np.random.rand(num_examples, self.hidden_num+1)
        hidden_states = hidden_states[:, 1:]
        return hidden_states

    '''
        :param hidden_data A matrix where each row consists of the states of the hidden units
        :return A matrix where each row consists of the visible units activated from the hidden
                units in the data matrix passed in.
    '''
    def get_visible_layer(self, hidden_data):
        num_examples = hidden_data.shape[0]
        visible_states = np.ones((num_examples, self.visible_num + 1))
        hidden_data = np.insert(hidden_data, 0, 1, axis=1)
        visible_activations = np.dot(hidden_data, self.weights.T)
        visible_probs = self.activation_logistic(visible_activations)
        visible_states[:, :] = visible_probs > np.random.rand(num_examples, self.visible_num+1)
        visible_states = visible_states[:, 1:]
        return visible_states

    '''
        :param visible visible layer
        :return v
    '''
    def predict(self, visible_data):
        num_example = visible_data.shape[0]
        hidden_states = np.ones((num_example, self.hidden_num+1))
        visible_data = np.insert(visible_data, 0, 1, axis=1)
        # forward
        hidden_activations = np.dot(visible_data, self.weights)
        hidden_probs = self.activation_logistic(hidden_activations)
        hidden_states[:, :] = hidden_probs > np.random.rand(num_example, self.hidden_num+1)
        # backward
        visible_states = np.ones((num_example, self.visible_num+1))
        visible_activations = np.dot(hidden_probs, self.weights.T)
        visible_probs = self.activation_logistic(visible_activations)
        return visible_probs[:, 1:]  # remove bias

if __name__ == '__main__':
    rbm = RBMModel(visible_num=6, hidden_num=2, learning_rate=0.1, min_batch=1000)
    train_set = np.array([[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [
                           0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0]])
    rbm.train(train_set, iters=1000, eps=1.0e-4)
    print 'weight:\n', rbm.weights
    rating = np.array([[0, 0, 0, 0.9, 0.7, 0]])
    hidden_data = rbm.get_hidden_layer(rating)
    print 'hidden_data:\n', hidden_data
    visible_data = rbm.get_visible_layer(hidden_data)
    print 'visible_data:\n', visible_data
    predict_data = rbm.predict(rating)
    print 'recommand score:'
    for i, score in enumerate(predict_data[0, :]):
        print i, score