import tensorflow as tf
from tensorflow import Variable
from tensorflow import placeholder
from network import graph


import sys
sys.path.append("game/")
import wrapped_flappy_bird as game

import random
import numpy as np
from collections import deque
from datetime import datetime
import cv2
import os

SAVE_DIR = 'network_state/'

GAMMA = 0.99                # decay rate of past observations

# before exploring
# OBSERVE = 100
OBSERVE = 1 * 10 ** 4
REPLAY_MEMORY = 5 * 10 ** 4

# EXPLOIT and EXPLORE
EPS_STEP = [0, 10 ** 6, 3 * 10 ** 6, 10 * 10 ** 6]
EPSILONs = [1, 0.1, 0.01, 0.001]

# Batch size
BATCH_SIZE = 32
FRAME_PER_ACTION = 1

# Saving
SAVING_PERIOD = 10 ** 5

TRAIN = False


def get_logname():
    return datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


def write_log(name, info):
    """
    :param name:    filename
    :param info:    list of info
    :return:
    """
    LOG_PATH = os.path.join('log/',  name)

    with open(LOG_PATH, 'a') as f:
        s = " ".join(map(
            lambda x: str(x), info))
        print s

        f.write(s + "\n")


def get_single_state(game_state, action):
    x_t, r_0, terminal = game_state.frame_step(action)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    return x_t, r_0, terminal


def train_dqn(s1, q1, s2, q2):
    # log
    log_name = get_logname()
    log_score = list()
    tot_score = 0

    # define network
    act = placeholder(tf.float32, [None, 2])
    y_ = placeholder(tf.float32, [None])

    y1 = tf.reduce_sum(tf.multiply(q1, act), reduction_indices=1)
    y2 = tf.reduce_sum(tf.multiply(q2, act), reduction_indices=1)

    loss1 = tf.reduce_mean(tf.square(y_ - y1))
    loss2 = tf.reduce_mean(tf.square(y_ - y2))

    trainer = tf.train.AdamOptimizer(1e-4)
    train_step1 = trainer.minimize(loss1)
    train_step2 = trainer.minimize(loss2)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(SAVE_DIR)

    game_state = game.GameState()

    # re-play memory D
    D = deque()

    # pre-processing
    pre_action = np.zeros(2)
    pre_action[0] = 1   # do nothing
    x_t, r_, t_ = get_single_state(game_state, pre_action)
    s_t0 = np.stack((x_t, x_t, x_t, x_t), axis=2)
    s_t = s_t0

    with tf.Session() as sess:
        sess.run(init)

        if TRAIN:
            print "Train Mode"
        else:
            print "Demo Mode"

        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print 'Successful loaded'
        else:
            print 'No model found'

        eps = EPSILONs[0]
        t = 0
        while True:

            q_choice = (random.random() > 0.5)

            if q_choice:
                q_t = q1.eval(feed_dict={
                    s1: [s_t]
                })[0]
            else:
                q_t = q2.eval(feed_dict={
                    s2: [s_t]
                })

            a_t = np.zeros([2])
            action_idx = 0

            if not TRAIN:
                eps = -1
            if random.random() <= eps:
                # print eps, True,
                action_idx = int(random.random() > 0.9)
            else:
                action_idx = np.argmax(q_t)
            a_t[action_idx] = 1

            # print a_t
            eps_i = 0
            for thres in EPS_STEP:
                if t > thres:
                    eps_i += 1

            # print t, eps_i
            Leps = EPSILONs[eps_i - 1]
            Heps = EPSILONs[eps_i]
            Lstp = EPS_STEP[eps_i - 1]
            Hstp = EPS_STEP[eps_i]
            eps += (Heps - Leps) / (Hstp - Lstp)
            # print eps

            x_t1, r_t, terminal = get_single_state(game_state, a_t)
            x_t1 = np.reshape(x_t1, (80, 80, 1))

            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
            D.append((s_t, a_t, r_t, s_t1, terminal))

            if not terminal:
                s_t = s_t1
            else:
                s_t = s_t0
            t += 1

            if not terminal:
                tot_score += r_t
            else:
                log_score.append(
                    np.array(tot_score).sum())
                tot_score = 0

            if len(D) > REPLAY_MEMORY:
                D.popleft()

            if t > OBSERVE and TRAIN:
                mini_batch = random.sample(D, BATCH_SIZE)

                s_t_batch = [d[0] for d in mini_batch]
                a_batch = [d[1] for d in mini_batch]
                r_batch = [d[2] for d in mini_batch]
                s_t1_batch = [d[3] for d in mini_batch]

                y_batch = []
                q1_batch = q1.eval(feed_dict={
                        s1: s_t1_batch})

                q2_batch = q2.eval(feed_dict={
                        s2: s_t1_batch})

                for i, d in enumerate(mini_batch):
                    if d[4]:
                        y_batch.append(r_batch[i])
                    else:
                        q_choice = (random.random() > 0.5)
                        if q_choice:
                            tmp_idx = np.argmax(q1_batch[i])
                            y_batch.append(r_batch[i] + GAMMA * q2_batch[i][tmp_idx])
                        else:
                            tmp_idx = np.argmax(q2_batch[i])
                            y_batch.append(r_batch[i] + GAMMA * q1_batch[i][tmp_idx])

                train_step1.run(feed_dict={
                    s1: s_t_batch,
                    act: a_batch,
                    y_: y_batch})

                train_step2.run(feed_dict={
                    s2: s_t_batch,
                    act: a_batch,
                    y_: y_batch})

                if not t % 100:
                    info = [
                        t, eps,
                        np.array(log_score).mean(),
                        loss1.eval(feed_dict={
                            s1: s_t_batch,
                            act: a_batch,
                            y_: y_batch}),
                        loss2.eval(feed_dict={
                            s2: s_t_batch,
                            act: a_batch,
                            y_: y_batch})]
                    write_log(log_name, info)
                    log_score = list()

            if not t % SAVING_PERIOD:
                saver.save(sess, save_path=SAVE_DIR, global_step=t)

if __name__ == "__main__":
    s1, q1 = graph()
    s2, q2 = graph()

    train_dqn(s1, q1, s2, q2)



    # if not t % 1:
    #     state = ""
    #     if t <= OBSERVE:
    #         state = "observe"
    #     elif t > OBSERVE and t <= OBSERVE + EXPLORE:
    #         state = "explore"
    #     else:
    #         state = "train"
    #
    #     print("TIMESTEP", t, "/ STATE", state,
    #           "/ EPS", eps,
    #           "/ ACTION", action_idx, "/ REWARD", r_t,
    #           "/ Q_MAX %e" % np.max(q_t))
