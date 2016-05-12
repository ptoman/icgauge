import sys
import copy
import random
import numpy as np
from numpy import dot, outer
import utils
import tensorflow as tf
from tensorflow.models.rnn import rnn

config = {
    batch_size = 10,
    num_steps = 5,
    hidden_size = 10,
    vocab_size = 5,
    keep_prob = 0.5 #can be 1
}


class LstmNetwork:
    """
    Currently doesn't do what *we* want and instead exactly copies the tutorial at:
    https://www.tensorflow.org/versions/r0.8/tutorials/recurrent/index.html

    >> 'The data required for this tutorial is in the data/ directory of the PTB dataset
    from Tomas  Mikolov's webpage: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

    The dataset is already preprocessed and contains overall 10000 different words, including 
    the end-of-sentence marker and a special symbol (<unk>) for rare words. We convert all of 
    them in the reader.py to unique integer identifiers to make it easy for the neural network 
    to process.'<<

    """
    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        if is_training and config.keep_prob < 1:
              lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                  lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size])
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        inputs = [tf.squeeze(input_, [1])
                  for input_ in tf.split(1, num_steps, inputs)]
        outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)

        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return


        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))


    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

def run_epoch(session, m, data, eval_op, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                      m.num_steps)):
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
        costs += cost
        iters += m.num_steps

        if verbose and step % (epoch_size // 10) == 10:
             print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                  iters * m.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _ = raw_data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                     config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config)
            mtest = PTBModel(is_training=False, config=eval_config)

        tf.initialize_all_variables().run()

        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                   verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
        print("Test Perplexity: %.3f" % test_perplexity)

if __name__ == "__main__":
    tf.app.run()





class TfShallowNeuralNetwork:
    """Fairly exact reproduction of `ShallowNeuralNetwork` in
    TensorFlow, differing only in some details of optimization.
    From Chris Potts in CS224U repository."""
    def __init__(self, hidden_dim=40, maxiter=100, eta=0.05):
        """All the parameters are set as attributes.
        
        Parameters
        ----------
        hidden_dim : int (default: 40)
            Dimensionality of the hidden layer.                   
        maxiter : int default: 100)
            Maximum number of training epochs.
            
        eta : float (default: 0.05)
            Learning rate.                 
        
        """
        self.input_dim = None
        self.hidden_dim = hidden_dim
        self.output_dim = None
        self.maxiter = maxiter
        self.eta = eta            
                
    def fit(self, training_data):
        """The training algorithm. 
        
        Parameters
        ----------
        training_data : list
            A list of (example, label) pairs, where `example`
            and `label` are both np.array instances.
        
        Attributes
        ----------
        self.sess : the TensorFlow session
        self.x : place holder for input data
        self.h : the hidden layer
        self.y : the output layer -- more like the full model here.
        self.W1 : dense weight connection from self.x to self.h
        self.b1 : bias
        self.W2 : dense weight connection from self.h to self.y
        self.b2 : bias
        self.y_ : placeholder for training data
                
        """
        self.sess = tf.InteractiveSession()
        # Dimensions determined by the data:
        self.input_dim = len(training_data[0][0])
        self.output_dim = len(training_data[0][1])        
        # Network initialization. For the inputs x, None in the first
        # dimension allows us to train and evaluate on datasets
        # of different size.
        self.x = tf.placeholder(tf.float32, [None, self.input_dim])
        self.W1 = tf.Variable(tf.random_normal([self.input_dim, self.hidden_dim]))
        self.b1 = tf.Variable(tf.random_normal([self.hidden_dim]))
        self.W2 = tf.Variable(tf.random_normal([self.hidden_dim, self.output_dim]))
        self.b2 = tf.Variable(tf.random_normal([self.output_dim]))
        # Network structure. As before, we use tanh for both 
        # layers. This is not strictly necessary, and TensorFlow
        # makes it easier to try different combinations.
        self.h = tf.nn.sigmoid(tf.matmul(self.x, self.W1) + self.b1)    
        self.y = tf.nn.sigmoid(tf.matmul(self.h, self.W2) + self.b2)        
        # A place holder for the true labels. None in the first
        # dimension allows us to train and evaluate on datasets
        # of different size.
        self.y_ = tf.placeholder(tf.float32, [None, self.output_dim])
        # This defines the objective as one of reducing the 
        # one-half squared total error. This could easily 
        # be made into a user-supplied parameter to facilitate
        # exploration of other costs. See
        # https://www.tensorflow.org/versions/r0.7/api_docs/python/math_ops.html#reduction
        cost = tf.reduce_sum(0.5 * (self.y_ - self.y)**tf.constant(2.0))
        # Simple GradientDescent (as opposed to the stochastic version
        # used by `ShallowNeuralNetwork`). For more options, see
        # https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#optimizers
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(cost)
        # TF session initialization:   
        tf.scalar_summary(cost.op.name, cost)
        init = tf.initialize_all_variables()
        self.sess.run(init)
        # Train (for larger datasets, the epochs should be batched):
        x, y_ = zip(*training_data)
        for iteration in range(self.maxiter):
            _, cost_val = self.sess.run([self.optimizer, cost], feed_dict={self.x: x, self.y_: y_})
            if iteration % 100 == 0:
                print "cost: %g" % cost_val

    def predict(self, ex):
        """
        Prediction for `ex`. This runs the model (forward propagation with
        self.x replaced by the single example `ex`).
        Parameters
        ----------
        ex : np.array
          Must be featurized as the training data were.
        Returns
        -------
        np.array
            The predicted outputs, dimension self.output_dim. TensorFlow
            assumes self.x is a list of examples and so returns a list of
            predictions. Since we're classifying just one, we return the
            list's only member.
            
        """
        return self.sess.run(self.y, feed_dict={self.x: [ex]})[0]


#if __name__ == '__main__':
#
#
#    def logical_operator_example(net):
#        train = [
#            # p  q    (p=q) (p v q)
#            ([1.,1.], [1.,   1.]), # T T ==> T, T
#            ([1.,0.], [0.,   1.]), # T F ==> F, T
#            ([0.,1.], [0.,   1.]), # F T ==> F, T
#            ([0.,0.], [1.,   0.])] # F F ==> T, F
#        net.fit(copy.deepcopy(train))        
#        for ex, labels in train:
#            prediction = net.predict(ex)
#            print(ex, labels, np.round(prediction, 2))
#
#    print('TensorFlow')
#    logical_operator_example(TfShallowNeuralNetwork(hidden_dim=4, eta=0.001, maxiter=6000))
