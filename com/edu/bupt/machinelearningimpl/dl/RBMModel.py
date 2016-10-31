
import numpy as np


class RBMModel:
    def __init__(self):
        pass

    def __init__(self, visible_num, hidden_num, learning_rate=0.1, min_batch=100):
        self.visible_num = visible_num
        self.hidden_num = hidden_num
        self.learning_rate = learning_rate
        self.min_batch = min_batch
        self.weights = 0.1 * np.random.randn(self.visible_num, self.hidden_num)
        self.weights = np.insert(self.weights, 0, 0, axis=0)
        self.weights = np.insert(self.weights, 0, 0, axis=1)

    def activation_logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def train(self, train_set, iters=1000, eps=1.0e-4):
        for step in xrange(iters):
            error = 0.0
            for i in xrange(0, train_set.shape[0], self.min_batch):
                num_examples = min(self.min_batch, train_set.shape[0]-i)
                data = train_set[i: i+num_examples, :]
                data = np.insert(data, 0, 1, axis=1)

                pos_hidden_activations = np.dot(data, self.weights)
                pos_hidden_probs = self.activation_logistic(pos_hidden_activations)
                pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.hidden_num+1)

                pos_associations = np.dot(data.T, pos_hidden_states)
                # pos_associations = np.dot(data.T, pos_hidden_probs)

                neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
                neg_visible_probs = self.activation_logistic(neg_visible_activations)
                neg_visible_probs[: 0] = 1

                neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
                neg_hidden_probs = self.activation_logistic(neg_hidden_activations)
                neg_hidden_states = neg_hidden_probs > np.random.rand(num_examples, self.hidden_num+1)

                neg_associations = np.dot(neg_visible_probs.T, neg_hidden_states)

                # neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

                self.weights += self.learning_rate * (pos_associations - neg_associations) / num_examples

                error += np.sum((data - neg_visible_probs)**2)
            if error < eps:
                break
            print 'iteration %d, error is %f' % (step, error)

    def get_hidden_layer(self, visible_data):
        num_examples = visible_data.shape[0]
        hidden_states = np.ones((num_examples, self.hidden_num + 1))
        visible_data = np.insert(visible_data, 0, 1, axis=1)
        hidden_activations = np.dot(visible_data, self.weights)
        hidden_probs = self.activation_logistic(hidden_activations)
        hidden_states[:, :] = hidden_probs > np.random.rand(num_examples, self.hidden_num+1)
        hidden_states = hidden_states[:, 1:]
        return hidden_states

    def get_visible_layer(self, hidden_data):
        num_examples = hidden_data.shape[0]
        visible_states = np.ones((num_examples, self.visible_num + 1))
        hidden_data = np.insert(hidden_data, 0, 1, axis=1)
        visible_activations = np.dot(hidden_data, self.weights.T)
        visible_probs = self.activation_logistic(visible_activations)
        visible_states[:, :] = visible_probs > np.random.rand(num_examples, self.visible_num+1)
        visible_states = visible_states[:, 1:]
        return visible_states

    def predict(self, visible_data):
        num_example = visible_data.shape[0]
        hidden_states = np.ones((num_example, self.hidden_num+1))
        visible_data = np.insert(visible_data, 0, 1, axis=1)

        hidden_activations = np.dot(visible_data, self.weights)
        hidden_probs = self.activation_logistic(hidden_activations)
        hidden_states[:, :] = hidden_probs > np.random.rand(num_example, self.hidden_num+1)

        visible_states = np.ones((num_example, self.visible_num+1))
        visible_activations = np.dot(hidden_probs, self.weights.T)
        visible_probs = self.activation_logistic(visible_activations)
        return visible_probs[:, 1:]

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