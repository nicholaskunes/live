#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Bayesian regression for latent source model and Bitcoin.

This module implements the 'Bayesian regression for latent source model' method
for predicting price variation of Bitcoin. You can read more about the method
at https://arxiv.org/pdf/1410.1231.pdf.
"""

import numpy as np
import bigfloat as bg
import time
from numpy.linalg import norm
from sklearn import linear_model
from sklearn.cluster import KMeans


def generate_timeseries(prices, n):
    m = len(prices) - n
    ts = np.empty((m, n + 1))
    for i in range(m):
        ts[i, :n] = prices[i:i + n]
        ts[i, n] = prices[i + n] - prices[i + n - 1]
    return ts


def find_cluster_centers(timeseries, k):
    k_means = KMeans(n_clusters=k)
    k_means.fit(timeseries)
    return k_means.cluster_centers_


def choose_effective_centers(centers, n):
    return centers[np.argsort(np.ptp(centers, axis=1))[-n:]]


def predict_dpi(x, s):
    num = 0
    den = 0
    for i in range(len(s)):
        y_i = s[i, len(x)]
        x_i = s[i, :len(x)]
        exp = bg.exp(-0.25 * norm(x - x_i) ** 2)
        num += y_i * exp
        den += exp
    return num / den


def linear_regression_vars(
    prices,
    v_bid,
    v_ask,
    s1,
    s2,
    s3,
    ):
    X = np.empty((len(prices) - 721, 4))
    Y = np.empty(len(prices) - 721)
    for i in range(720, len(prices) - 1):
        dp = prices[i + 1] - prices[i]
        dp1 = predict_dpi(prices[i - 180:i], s1)
        dp2 = predict_dpi(prices[i - 360:i], s2)
        dp3 = predict_dpi(prices[i - 720:i], s3)
        r = (v_bid[i] - v_ask[i]) / (v_bid[i] + v_ask[i])
        X[i - 720, :] = [dp1, dp2, dp3, r]
        Y[i - 720] = dp
    return (X, Y)


def find_parameters_w(X, Y):
    clf = linear_model.LinearRegression()
    clf.fit(X, Y)
    w0 = clf.intercept_
    (w1, w2, w3, w4) = clf.coef_
    return (w0, w1, w2, w3, w4)


def predict_dps(
    prices,
    v_bid,
    v_ask,
    s1,
    s2,
    s3,
    w,
    ):
    dps = []
    (w0, w1, w2, w3, w4) = w
    for i in range(720, len(prices) - 2):
        dp1 = predict_dpi(prices[i - 180:i], s1)
        dp2 = predict_dpi(prices[i - 360:i], s2)
        dp3 = predict_dpi(prices[i - 720:i], s3)
        r = (v_bid[i] - v_ask[i]) / (v_bid[i] + v_ask[i])
        dp = w0 + w1 * dp1 + w2 * dp2 + w3 * dp3 + w4 * r
        prices[i + 1] = prices[i] + dp
        dps.append(float(dp))
    return dps


def evaluate_performance(
    prices,
    dps,
    t,
    step,
    ):
    bank_balance = 0
    fees_paid = 0
    position = 0
    trade_count = 0
    prev_pos = 'NONE'
    prior_value = prices[720]

    print '[bayesian regression clustered prediction algorithm]'
    print '---------- [def key]'
    print "---------- [    INSIG] INSIGNIFICANCE: this flag is assigned if PCS < 0.00075, its complement is 'SIGNIF'"
    print '---------- [    ITR] ITERATION: the cur. sequence of the iterator'
    print '---------- [    PPB] POSTED PRIOR BALANCE: the prev. balance prior to the the cur. trade'
    print '---------- [    CPB] CUR. POSTED BALANCE: the cur. balance subsequent to the the cur. trade'
    print '---------- [    CTV] CUR. TRADE VALUE: the cur. trade currency value'
    print '---------- [    PCD] PRICE CHANGE DIFF.: the magnitude of change between the prev. trade value and the cur. [PCD = abs(PTV - CTV)]'
    print '---------- [    PCS] PRICE CHANGE SIGNIFICANCE: the significance of the PCD [PCS = (PCD / CTV)]'
    print '---------- [    PTV] PREV. TRADE VALUE: the prev. trade currency value'
    print '---------- [    PTP] PREV. TRADE POSITION: the prev. trade position (e.g. LONG or SHORT)'
    print '---------- [    CTF] TRADE FEE DIFF.: the prev. position trade fee added to the cur. to-be-executed trade fee'
    print '---------- [end key]'

    for i in range(720, 720 + 1080, step):

        # long position - BUY

        print str(dps[i - 720]) + ', ' + str(prices[i])
        if dps[i - 720] > t and position <= 0:
            if abs(float(prior_value) - float(prices[i])) \
                / float(prices[i]) < 0.00075 and prev_pos != 'LONG':
                continue
            position += 1
            prior_balance = bank_balance
            bank_balance -= prices[i] + prices[i] * 0.00075
            fees_paid += prices[i] * 0.00075
            trade_count += 1
            insig = 'SIGNIF'
            if abs(float(prior_value) - float(prices[i])) \
                / float(prices[i]) < 0.00075:
                insig = 'INSIG'
            prev_pos = 'LONG'
            prior_value = prices[i]

    # short position - SELL

        if dps[i - 720] < -t and position >= 0:
            if abs(float(prior_value) - float(prices[i])) \
                / float(prices[i]) < 0.00075 and prev_pos != 'SHORT':
                continue
            position -= 1
            prior_balance = bank_balance
            bank_balance += prices[i] - prices[i] * 0.00075
            fees_paid += prices[i] * 0.00075
            trade_count += 1
            insig = 'SIGNIF'
            if abs(float(prior_value) - float(prices[i])) \
                / float(prices[i]) < 0.00075:
                insig = 'INSIG'
            prev_pos = 'SHORT'
            prior_value = prices[i]

    # sell what you bought

    if position == 1:
        bank_balance += prices[len(prices) - 1]

    # pay back what you borrowed

    if position == -1:
        bank_balance -= prices[len(prices) - 1]
    print '[series statistics ' + '\n    [length]  ' + str((i - 720)
            * 10 / 60) + ' min.\n    [profit] $' + str(bank_balance) \
        + '\n    [fees paid] $' + str(fees_paid) + '\n    [revenue] $' \
        + str(fees_paid + bank_balance) + '\n    [trades exec.] ' \
        + str(trade_count) + '''
    ]
'''

    return bank_balance
