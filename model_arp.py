import numpy as np
import tensorflow as tf
import sys

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.python.framework import ops
from functools import partial
from utils import *

def run_with_hyperdict(hyperdict, fold, live = False):
    dict_train  = np.load('__dataset__/5_iterate/train_' + str(fold + 1) + '.npy')[()]
    dict_valid  = np.load('__dataset__/5_iterate/valid_' + str(fold + 1) + '.npy')[()]
    dict_test   = np.load('__dataset__/5_iterate/test_'  + str(fold + 1) + '.npy')[()]

    limit_iter = hyperdict['limit_iter']
    limit_plat = hyperdict['limit_plat']
    learn_rate = hyperdict['learn_rate']
    early_rate = hyperdict['early_rate']
    l2_penalty = hyperdict['l2_penalty']
    state_size = hyperdict['state_size']
    batch_size = hyperdict['batch_size']
    total_size = len(dict_train['x_stat'])

    x_stat_size = dict_train['x_stat'].shape[1]
    x_tcon_size = dict_train['x_tcon'].shape[2]
    x_tbin_size = dict_train['x_tbin'].shape[2]
    y_tcon_size = dict_train['y_tcon'].shape[2]
    y_tbin_size = dict_train['y_tbin'].shape[2]
    m_tcon_size = dict_train['m_tcon'].shape[2]
    m_tbin_size = dict_train['m_tbin'].shape[2]

    x_temp_size = x_tcon_size + x_tbin_size
    y_temp_size = y_tcon_size + y_tbin_size

    h_size = state_size * x_temp_size

    ops.reset_default_graph()

    teacher = tf.placeholder(tf.bool)

    x_stat = tf.placeholder(tf.float32, shape = [None,                   x_stat_size])
    x_tcon = tf.placeholder(tf.float32, shape = [None, MAX_PRE,          x_tcon_size])
    x_tbin = tf.placeholder(tf.float32, shape = [None, MAX_PRE,          x_tbin_size])
    y_tcon = tf.placeholder(tf.float32, shape = [None,          MAX_TAU, y_tcon_size])
    y_tbin = tf.placeholder(tf.float32, shape = [None,          MAX_TAU, y_tbin_size])
    m_tcon = tf.placeholder(tf.  int32, shape = [None,          MAX_TAU, m_tcon_size])
    m_tbin = tf.placeholder(tf.  int32, shape = [None,          MAX_TAU, m_tbin_size])

    x_temp = tf.concat([x_tcon, x_tbin], axis = 2)
    y_temp = tf.concat([y_tcon, y_tbin], axis = 2)

    prelen = tf.placeholder(tf.  int32, shape = [None                               ])
    poslen = tf.placeholder(tf.  int32, shape = [None                               ])

    x = tf.concat([
        x_stat,
        tf.reshape(x_tcon, [-1, MAX_PRE * x_tcon_size]),
        tf.reshape(x_tbin, [-1, MAX_PRE * x_tbin_size]),
    ], axis = 1)

    y = tf.concat([
        tf.reshape(y_tcon, [-1, MAX_TAU * y_tcon_size]),
        tf.reshape(y_tbin, [-1, MAX_TAU * y_tbin_size]),
    ], axis = 1)

    x_size = x_stat_size + MAX_PRE * (x_tcon_size + x_tbin_size)
    y_size =               MAX_TAU * (y_tcon_size + y_tbin_size)

    Wp = tf.Variable(tf.truncated_normal([x_size, h_size], mean = 0, stddev = 0.01))
    bp = tf.Variable(tf.truncated_normal([        h_size], mean = 0, stddev = 0.01), name = 'bias')

    We = tf.Variable(tf.truncated_normal([y_size, h_size], mean = 0, stddev = 0.01))
    be = tf.Variable(tf.truncated_normal([        h_size], mean = 0, stddev = 0.01), name = 'bias')

    Wd = tf.Variable(tf.truncated_normal([h_size, y_size], mean = 0, stddev = 0.01))
    bd = tf.Variable(tf.truncated_normal([        y_size], mean = 0, stddev = 0.01), name = 'bias')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GRAPH
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    h_hat = tf.matmul(x, Wp)
    y_hat = tf.reshape(tf.matmul(h_hat, Wd), [-1, MAX_TAU, y_tcon_size + y_tbin_size])
    y_hat_tcon = y_hat[:, :, :y_tcon_size ]
    y_hat_tlog = y_hat[:, :,  y_tcon_size:]
    y_hat_tbin = tf.nn.sigmoid(y_hat_tlog)

    h_til = tf.matmul(y, We)
    y_til = tf.reshape(tf.matmul(h_til, Wd), [-1, MAX_TAU, y_tcon_size + y_tbin_size])
    y_til_tcon = y_til[:, :, :y_tcon_size ]
    y_til_tlog = y_til[:, :,  y_tcon_size:]
    y_til_tbin = tf.nn.sigmoid(y_til_tlog)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VARIABLES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    theta_predict = [Wp, bp]
    theta_encoder = [We, be]
    theta_decoder = [Wd, bd]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LOSSES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    FCUR_COL = 3
    FPRE_COL = 4

    loss_hat_tcon = [
        mse_tau(at(y_tcon, tau), at(y_hat_tcon, tau), at(m_tcon, tau))
        for tau in range(MAX_TAU)]

    loss_hat_tbin = [
        mce_tau(at(y_tbin, tau), at(y_hat_tlog, tau), at(m_tbin, tau))
        for tau in range(MAX_TAU)]

    loss_hat_fcur = [
        mse_tau_col(at(y_tcon, tau), at(y_hat_tcon, tau), at(m_tcon, tau), FCUR_COL)
        for tau in range(MAX_TAU)]

    loss_hat_fpre = [
        mse_tau_col(at(y_tcon, tau), at(y_hat_tcon, tau), at(m_tcon, tau), FPRE_COL)
        for tau in range(MAX_TAU)]

    loss_til_tcon = [
        mse_tau(at(y_tcon, tau), at(y_til_tcon, tau), at(m_tcon, tau))
        for tau in range(MAX_TAU)]

    loss_til_tbin = [
        mce_tau(at(y_tbin, tau), at(y_til_tlog, tau), at(m_tbin, tau))
        for tau in range(MAX_TAU)]

    loss_til_fcur = [
        mse_tau_col(at(y_tcon, tau), at(y_til_tcon, tau), at(m_tcon, tau), FCUR_COL)
        for tau in range(MAX_TAU)]

    loss_til_fpre = [
        mse_tau_col(at(y_tcon, tau), at(y_til_tcon, tau), at(m_tcon, tau), FPRE_COL)
        for tau in range(MAX_TAU)]

    loss_lat = mse_latent(h_til, h_hat)

    l2 = [tf.nn.l2_loss(tv)
            for tv in tf.trainable_variables() if not ('bias' in tv.name)]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SOLVERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    optimizer = tf.train.AdamOptimizer(learn_rate)

    solver_101 = optimizer.minimize(
        sum(loss_hat_tcon) +
        sum(loss_hat_tbin) +
        sum(l2) * l2_penalty,
        var_list = theta_predict + theta_decoder
    )

    solver_011 = optimizer.minimize(
        sum(loss_til_tcon) +
        sum(loss_til_tbin) +
        sum(l2) * l2_penalty,
        var_list = theta_encoder + theta_decoder
    )

    solver_100 = optimizer.minimize(
        loss_lat           +
        sum(l2) * l2_penalty,
        var_list = theta_predict
    )

    solver_010 = optimizer.minimize(
        loss_lat           +
        sum(l2) * l2_penalty,
        var_list = theta_encoder
    )

    solver_111 = optimizer.minimize(
        sum(loss_til_tcon) +
        sum(loss_til_tbin) +
        sum(loss_hat_tcon) +
        sum(loss_hat_tbin) +
        sum(l2) * l2_penalty,
        var_list = theta_predict + theta_encoder + theta_decoder
    )

    S_INF = lambda tau: np.s_[:, tau, :11 ]
    S_COM = lambda tau: np.s_[:, tau,  11:]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HELPERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_feed(dict, indices = slice(None), force = False):
        return {
            x_stat: dict['x_stat'][indices],
            x_tcon: dict['x_tcon'][indices],
            x_tbin: dict['x_tbin'][indices],
            y_tcon: dict['y_tcon'][indices],
            y_tbin: dict['y_tbin'][indices],
            m_tcon: dict['m_tcon'][indices],
            m_tbin: dict['m_tbin'][indices],
            prelen: dict['prelen'][indices],
            poslen: dict['poslen'][indices],

            teacher: force,
        }

    def show_run(
        run_loss_hat_tcon,
        run_loss_hat_tbin,
        run_loss_hat_fcur,
        run_loss_hat_fpre,
        run_roc_hat_inf  ,
        run_roc_hat_com  ,
        run_prc_hat_inf  ,
        run_prc_hat_com  ,

        run_loss_til_tcon,
        run_loss_til_tbin,
        run_loss_til_fcur,
        run_loss_til_fpre,
        run_roc_til_inf  ,
        run_roc_til_com  ,
        run_prc_til_inf  ,
        run_prc_til_com  ,
    ):
        DIVIDE = '------------------------------------------------------------------------------------------------'
        HEADER = '  steps   tau  loss_tcon  loss_tbin  loss_fcur  loss_fpre  roc_inf  roc_com   prc_inf  prc_com  '

        print(DIVIDE + '      ' + DIVIDE)
        print(HEADER + '      ' + HEADER)
        print(DIVIDE + '      ' + DIVIDE)

        for tau in range(MAX_TAU):
            print('%d     %0.3f      %0.3f      %0.3f   %8.3f     %.3f    %.3f     %.3f    %.3f   ' \
                % (tau + 1, run_loss_hat_tcon[tau], run_loss_hat_tbin[tau], run_loss_hat_fcur[tau], run_loss_hat_fpre[tau], run_roc_hat_inf[tau], run_roc_hat_com[tau], run_prc_hat_inf[tau], run_prc_hat_com[tau]), end = '      ')
            print('%d     %0.3f      %0.3f      %0.3f   %8.3f     %.3f    %.3f     %.3f    %.3f   ' \
                % (tau + 1, run_loss_til_tcon[tau], run_loss_til_tbin[tau], run_loss_til_fcur[tau], run_loss_til_fpre[tau], run_roc_til_inf[tau], run_roc_til_com[tau], run_prc_til_inf[tau], run_prc_til_com[tau]))

    def save_run(
        destination      ,

        run_loss_hat_tcon,
        run_loss_hat_tbin,
        run_loss_hat_fcur,
        run_loss_hat_fpre,
        run_roc_hat_inf  ,
        run_roc_hat_com  ,
        run_prc_hat_inf  ,
        run_prc_hat_com  ,

        run_loss_til_tcon,
        run_loss_til_tbin,
        run_loss_til_fcur,
        run_loss_til_fpre,
        run_roc_til_inf  ,
        run_roc_til_com  ,
        run_prc_til_inf  ,
        run_prc_til_com  ,
    ):
        destination.append({
            'hyperdict'     : hyperdict        ,

            'loss_hat_tcon' : run_loss_hat_tcon,
            'loss_hat_tbin' : run_loss_hat_tbin,
            'loss_hat_fcur' : run_loss_hat_fcur,
            'loss_hat_fpre' : run_loss_hat_fpre,
            'roc_hat_inf'   : run_roc_hat_inf  ,
            'roc_hat_com'   : run_roc_hat_com  ,
            'prc_hat_inf'   : run_prc_hat_inf  ,
            'prc_hat_com'   : run_prc_hat_com  ,

            'loss_til_tcon' : run_loss_til_tcon,
            'loss_til_tbin' : run_loss_til_tbin,
            'loss_til_fcur' : run_loss_til_fcur,
            'loss_til_fpre' : run_loss_til_fpre,
            'roc_til_inf'   : run_roc_til_inf  ,
            'roc_til_com'   : run_roc_til_com  ,
            'prc_til_inf'   : run_prc_til_inf  ,
            'prc_til_com'   : run_prc_til_com  ,
        })

    save_all = partial(save_run, ALL_COMBOS[fold])

    save_top = partial(save_run, TOP_COMBOS)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LEARNING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def do_train(feed_dict, solver):
        (
            _,
        ) = sess.run((
            solver,
        ), feed_dict = feed_dict)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INFERENCE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def do_infer(feed_dict, grid_func):
        (
            run_loss_hat_tcon,
            run_loss_hat_tbin,
            run_loss_hat_fcur,
            run_loss_hat_fpre,

            run_loss_til_tcon,
            run_loss_til_tbin,
            run_loss_til_fcur,
            run_loss_til_fpre,

            run_y_hat_tbin,
            run_y_til_tbin,
            run_y_tbin,
            run_m_tbin,
        ) = sess.run((
            loss_hat_tcon,
            loss_hat_tbin,
            loss_hat_fcur,
            loss_hat_fpre,

            loss_til_tcon,
            loss_til_tbin,
            loss_til_fcur,
            loss_til_fpre,

            y_hat_tbin,
            y_til_tbin,
            y_tbin,
            m_tbin,
        ), feed_dict = feed_dict)

        run_roc_hat_inf = [auc_roc(
            run_y_tbin[S_INF(tau)], run_y_hat_tbin[S_INF(tau)],
            run_m_tbin[S_INF(tau)]) for tau in range(MAX_TAU)]

        run_roc_hat_com = [auc_roc(
            run_y_tbin[S_COM(tau)], run_y_hat_tbin[S_COM(tau)],
            run_m_tbin[S_COM(tau)]) for tau in range(MAX_TAU)]

        run_prc_hat_inf = [auc_prc(
            run_y_tbin[S_INF(tau)], run_y_hat_tbin[S_INF(tau)],
            run_m_tbin[S_INF(tau)]) for tau in range(MAX_TAU)]

        run_prc_hat_com = [auc_prc(
            run_y_tbin[S_COM(tau)], run_y_hat_tbin[S_COM(tau)],
            run_m_tbin[S_COM(tau)]) for tau in range(MAX_TAU)]

        run_roc_til_inf = [auc_roc(
            run_y_tbin[S_INF(tau)], run_y_til_tbin[S_INF(tau)],
            run_m_tbin[S_INF(tau)]) for tau in range(MAX_TAU)]

        run_roc_til_com = [auc_roc(
            run_y_tbin[S_COM(tau)], run_y_til_tbin[S_COM(tau)],
            run_m_tbin[S_COM(tau)]) for tau in range(MAX_TAU)]

        run_prc_til_inf = [auc_prc(
            run_y_tbin[S_INF(tau)], run_y_til_tbin[S_INF(tau)],
            run_m_tbin[S_INF(tau)]) for tau in range(MAX_TAU)]

        run_prc_til_com = [auc_prc(
            run_y_tbin[S_COM(tau)], run_y_til_tbin[S_COM(tau)],
            run_m_tbin[S_COM(tau)]) for tau in range(MAX_TAU)]

        grid_func(
            run_loss_hat_tcon,
            run_loss_hat_tbin,
            run_loss_hat_fcur,
            run_loss_hat_fpre,
            run_roc_hat_inf  ,
            run_roc_hat_com  ,
            run_prc_hat_inf  ,
            run_prc_hat_com  ,

            run_loss_til_tcon,
            run_loss_til_tbin,
            run_loss_til_fcur,
            run_loss_til_fpre,
            run_roc_til_inf  ,
            run_roc_til_com  ,
            run_prc_til_inf  ,
            run_prc_til_com  ,
        )

        return \
            sum(run_loss_hat_tcon) + sum(run_loss_hat_tbin), \
            sum(run_loss_til_tcon) + sum(run_loss_til_tbin)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    disk = tf.train.Saver()

    SOLVERS = [
        solver_011,
        solver_100,
        solver_111,
    ]

    SCORING = [
        [   1],
        [0   ],
        [0   ],
    ]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EARLY STOP
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    phase = 0

    while phase < 3:
        best_score = float('inf')
        plat_count = 0

        for i in range(limit_iter):
            if i % early_rate == 0:
                print('  %5d    ' % (i), end = '')

                results = do_infer(get_feed(dict_valid), show_run)
                score = sum([results[s] for s in SCORING[phase]])

                if score < best_score:
                    best_score = score
                    disk.save(sess, '.tmp_arp/model.ckpt')

                    plat_count = 0
                    print('<')
                else:
                    plat_count = plat_count + 1
                    print(' ')

                    if plat_count > limit_plat:
                        break

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TRAINING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            indices = random_batch_indices(total_size, batch_size)
            do_train(get_feed(dict_train, indices = indices), SOLVERS[phase])

        phase = phase + 1
        disk.restore(sess, '.tmp_arp/model.ckpt')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VALIDATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if not live:
        do_infer(get_feed(dict_valid), save_all)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TESTING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    else:
        do_infer(get_feed(dict_test ), save_top)

best_hyperdict = np.load('.tmp_arp/best_hyperdict.npy')[()]

for fold in range(FOLDS):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('FOLD %d of %d' % (fold + 1, FOLDS))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    run_with_hyperdict(best_hyperdict, fold, live = True)

np.save('__results__/arp_' + str(SAMPLE) + '_' + str(MAX_TAU) + '.npy', TOP_COMBOS)
