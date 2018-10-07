import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import gym
import cPickle as pickle


def preprocess(I):
  """ preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195]                                             # crop the image borders
  I = I[::2,::2,0]                                          # downsample by factor of 2
  I[I == 144] = 0                                           # erase background (background type 1)
  I[I == 109] = 0                                           # erase background (background type 2)
  I[I != 0] = 1                                             # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def init_weights_w_1(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    if resume:
        return tf.Variable(model['w_1'])
    return tf.Variable(weights)

def init_weights_w_2(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    if resume:
        return tf.Variable(model['w_2'])
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    layer1 = tf.nn.relu(tf.matmul(X, w_1))      
    output = tf.matmul(layer1, w_2)                                
    return output

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_reward = np.zeros_like(r)
  running_score = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_score = 0                         # reset the sum, since this was a game boundary (pong specific!)
    running_score = running_score * 0.99 + r[t]
    discounted_reward[t] = running_score
  return discounted_reward


def main():
    D = 80 * 80                                             # input dimensionality: 80x80 grid
    env = gym.make("Pong-v0")
    observation = env.reset()
    prev_x = None                                           # used in computing the difference frame
    x_list = []
    reward_list = []
    action_list = []
    global episode_number, resume, model

    model = {}
    render = False
    resume = False
    episode_number = 0

    if resume:
        model = pickle.load(open('tf_model_old', 'rb'))
        episode_number = model['episode_number']

    # Layer's sizes
    x_size = 6400                                           # Number of input nodes
    h_size = 256                                            # Number of hidden nodes
    y_size = 1                                              # Number of outcomes 

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    Y = tf.placeholder("float", shape=[None, y_size])
    discounted_rewards = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights_w_1((x_size, h_size))
    w_2 = init_weights_w_2((h_size, y_size))

    # Forward propagation
    Y_pred    = forwardprop(X, w_1, w_2)
    action_prob = tf.sigmoid(Y_pred)

    # Backward propagation
    cost    = tf.matmul(discounted_rewards, tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_pred), transpose_a=True)
    updates = tf.train.RMSPropOptimizer(5e-4, momentum= 0.5,epsilon=1e-10).minimize(cost)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    reward_sum = 0.0
    running_reward = 0.0
    opp_score = 0.0
    our_score = 0.0
    batch = 1

    while True:
        if render: env.render()
        # preprocess the observation, set input to network to be difference image
        cur_x = preprocess(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        x_list.append(x)
        prev_x = cur_x
        action = 2 if np.random.uniform() < sess.run(action_prob, feed_dict={X:x.reshape((1,6400))}) else 3         # roll the dice!
        action_list.append(3-action)
        observation, reward, done, info = env.step(action)
        reward_list.append(reward)
        reward_sum += reward
        
        # Train with each example
        if done: # an episode finished
            episode_number += 1
            observation = env.reset()                                                                               # reset env
            prev_x = None
            reward_vector = np.asarray(reward_list, dtype=np.float32)
            running_reward = reward_sum if running_reward==0 else running_reward * 0.99 + reward_sum * 0.01
            reward_sum = 0.0

            if episode_number%batch == 0:
                sess.run(updates, feed_dict={X: np.vstack(x_list), Y: np.vstack(action_list), discounted_rewards: np.vstack(discount_rewards(reward_vector))})
                if episode_number%128 == 0:
                    print  episode_number,": %.3f : %.3f : %.3f" %(opp_score/(opp_score+our_score), our_score/(our_score+opp_score), running_reward) , ">>" , len(x_list)
                    model['w_1'] = sess.run(w_1)
                    model['w_2'] = sess.run(w_2)
                    model['episode_number'] = episode_number
                    model_f = open('tf_model_new', 'wb')
                    pickle.dump(model, model_f)
                    model_f.close()
                    opp_score = 0.0
                    our_score = 0.0
                if len(x_list) < 4000:
                    batch = 2*batch
                if len(x_list) > 8000 and batch > 1:
                    batch = batch/2
                opp_score += reward_list.count(-1)
                our_score += reward_list.count(+1)
                x_list = []
                reward_list = []
                action_list = []
                if running_reward > 1.0:
                    break;

    sess.close()

if __name__ == '__main__':
    main()
