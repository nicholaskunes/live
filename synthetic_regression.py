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
    """Use the first time period to generate all possible time series of length n
       and their corresponding label.

    Args:
        prices: A numpy array of floats representing prices over the first time
            period.
        n: An integer (180, 360, or 720) representing the length of time series.

    Returns:
        A 2-dimensional numpy array of size (len(prices)-n) x (n+1). Each row
        represents a time series of length n and its corresponding label
        (n+1-th column).
    """
    m = len(prices) - n
    ts = np.empty((m, n + 1))
    for i in range(m):
        ts[i, :n] = prices[i:i + n]
        ts[i, n] = prices[i + n] - prices[i + n - 1]
    return ts


def find_cluster_centers(timeseries, k):
    """Cluster timeseries in k clusters using k-means and return k cluster centers.

    Args:
        timeseries: A 2-dimensional numpy array generated by generate_timeseries().
        k: An integer representing the number of centers (e.g. 100).

    Returns:
        A 2-dimensional numpy array of size k x num_columns(timeseries). Each
        row represents a cluster center.
    """
    k_means = KMeans(n_clusters=k)
    k_means.fit(timeseries)
    return k_means.cluster_centers_


def choose_effective_centers(centers, n):
    """Choose n most effective cluster centers with high price variation."""
    return centers[np.argsort(np.ptp(centers, axis=1))[-n:]]


def predict_dpi(x, s):
    """Predict the average price change Δp_i, 1 <= i <= 3.

    Args:
        x: A numpy array of floats representing previous 180, 360, or 720 prices.
        s: A 2-dimensional numpy array generated by choose_effective_centers().

    Returns:
        A big float representing average price change Δp_i.
    """
    num = 0
    den = 0
    for i in range(len(s)):
        y_i = s[i, len(x)]
        x_i = s[i, :len(x)]
        exp = bg.exp(-0.25 * norm(x - x_i) ** 2)
        num += y_i * exp
        den += exp
    return num / den


def linear_regression_vars(prices, v_bid, v_ask, s1, s2, s3):
    """Use the second time period to generate the independent and dependent variables
       in the linear regression model Δp = w0 + w1 * Δp1 + w2 * Δp2 + w3 * Δp3 + w4 * r.

    Args:
        prices: A numpy array of floats representing prices over the second time
            period.
        v_bid: A numpy array of floats representing total volumes people are
            willing to buy over the second time period.
        v_ask: A numpy array of floats representing total volumes people are
            willing to sell over the second time period.
        s1: A 2-dimensional numpy array generated by choose_effective_centers()
        s2: A 2-dimensional numpy array generated by choose_effective_centers().
        s3: A 2-dimensional numpy array generated by choose_effective_centers().

    Returns:
        A tuple (X, Y) representing the independent and dependent variables in
        the linear regression model. X is a 2-dimensional numpy array and each
        row represents [Δp1, Δp2, Δp3, r]. Y is a numpy array of floats and
        each array element represents Δp.
    """
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
    return X, Y


def find_parameters_w(X, Y):
    """Find the parameter values w for the model which best fits X and Y.

    Args:
        X: A 2-dimensional numpy array representing the independent variables
            in the linear regression model.
        Y: A numpy array of floats representing the dependent variables in the
            linear regression model.

    Returns:
        A tuple (w0, w1, w2, w3, w4) representing the parameter values w.
    """
    clf = linear_model.LinearRegression()
    clf.fit(X, Y)
    w0 = clf.intercept_
    w1, w2, w3, w4 = clf.coef_
    return w0, w1, w2, w3, w4


def predict_dps(prices, v_bid, v_ask, s1, s2, s3, w):
    """Predict average price changes (final estimations Δp) over the third
       time period.

    Args:
        prices: A numpy array of floats representing prices over the third time
            period.
        v_bid: A numpy array of floats representing total volumes people are
            willing to buy over the third time period.
        v_ask: A numpy array of floats representing total volumes people are
            willing to sell over the third time period.
        s1: A 2-dimensional numpy array generated by choose_effective_centers()
        s2: A 2-dimensional numpy array generated by choose_effective_centers().
        s3: A 2-dimensional numpy array generated by choose_effective_centers().
        w: A tuple (w0, w1, w2, w3, w4) generated by find_parameters_w().

    Returns:
        A numpy array of floats. Each array element represents the final
        estimation Δp.
    """
    dps = []
    w0, w1, w2, w3, w4 = w
    for i in range(720, len(prices) - 1):
        dp1 = predict_dpi(prices[i - 180:i], s1)
        dp2 = predict_dpi(prices[i - 360:i], s2)
        dp3 = predict_dpi(prices[i - 720:i], s3)
        r = (v_bid[i] - v_ask[i]) / (v_bid[i] + v_ask[i])
        dp = w0 + w1 * dp1 + w2 * dp2 + w3 * dp3 + w4 * r
        dps.append(float(dp))
    return dps

def evaluate_performance(prices, dps, t, step):
    """Use the third time period to evaluate the performance of the algorithm.

    Args:
        prices: A numpy array of floats representing prices over the third time
            period.
        dps: A numpy array of floats generated by predict_dps().
        t: A number representing a threshold.
        step: An integer representing time steps (when we make trading decisions).

    Returns:
        A number representing the bank balance.
    """
    bank_balance = 0
    fees_paid = 0
    position = 0
    trade_count = 0
    prev_pos = "NONE"
    prior_value = prices[720]
	
    print "[bayesian regression clustered prediction algorithm]"
    print "---------- [def key]"
    print "---------- [    INSIG] INSIGNIFICANCE: this flag is assigned if PCS < 0.00075, its complement is 'SIGNIF'"
    print "---------- [    ITR] ITERATION: the cur. sequence of the iterator"
    print "---------- [    PPB] POSTED PRIOR BALANCE: the prev. balance prior to the the cur. trade"
    print "---------- [    CPB] CUR. POSTED BALANCE: the cur. balance subsequent to the the cur. trade"
    print "---------- [    CTV] CUR. TRADE VALUE: the cur. trade currency value"
    print "---------- [    PCD] PRICE CHANGE DIFF.: the magnitude of change between the prev. trade value and the cur. [PCD = abs(PTV - CTV)]"
    print "---------- [    PCS] PRICE CHANGE SIGNIFICANCE: the significance of the PCD [PCS = (PCD / CTV)]"
    print "---------- [    PTV] PREV. TRADE VALUE: the prev. trade currency value"
    print "---------- [    PTP] PREV. TRADE POSITION: the prev. trade position (e.g. LONG or SHORT)"
    print "---------- [    CTF] TRADE FEE DIFF.: the prev. position trade fee added to the cur. to-be-executed trade fee"
    print "---------- [end key]"
	
    for i in range(720, len(prices) - 1, step):
        # long position - BUY
	print(dps[i-720] + ", " + prices[i])
        if dps[i - 720] > t and position <= 0:
	    if ((abs(float(prior_value) - float(prices[i])) / float(prices[i]))) < 0.00075 and prev_pos != "LONG":
	   	 print(
	        	#"[synthetic LONG INSIG GATE-1\n"
	        	#"    [ITR]  " + str(i - 720) + "\n"
	        	#"    [PCS]  " + str(((abs(float(prior_value) - float(prices[i])) / float(prices[i])))) + "\n"
	        	#"    [PTP]  " + str(prev_pos) + "\n"
			#"    ]\n"
	    	 )
		 continue
            position += 1
            prior_balance = bank_balance
            bank_balance -= (prices[i] + (prices[i] * 0.00075))
	    fees_paid += (prices[i] * 0.00075)
	    trade_count += 1
	    insig = "SIGNIF"
	    if ((abs(float(prior_value) - float(prices[i])) / float(prices[i]))) < 0.00075:
		insig = "INSIG"
	    print(
	        "[synthetic LONG " + insig + "\n"
	        #"    [ITR]  " + str(i - 720) + "\n"
	        #"    [PPB] $" + str(prior_balance) + "\n"
	        #"    [PTV] $" + str(prior_value) + "\n"
	        #"    [CTV] $" + str(prices[i]) + "\n"
	        #"    [PCD] $" + str(abs(float(prior_value) - float(prices[i]))) + "\n"
	        #"    [PCS]  " + str(((abs(float(prior_value) - float(prices[i])) / float(prices[i])))) + "\n"
	        #"    [PTP]  " + str(prev_pos) + "\n"
	        #"    [CPB] $" + str(bank_balance) + "\n"
		#"    ]\n"
	    )
	    prev_pos = "LONG"
	    prior_value = prices[i]
	# short position - SELL
        if dps[i - 720] < -t and position >= 0:
	    if ((abs(float(prior_value) - float(prices[i])) / float(prices[i]))) < 0.00075 and prev_pos != "SHORT":
	   	 print(
	        	#"[synthetic SHORT INSIG GATE-1\n"
	        	#"    [ITR]  " + str(i - 720) + "\n"
	        	#"    [PCS]  " + str(((abs(float(prior_value) - float(prices[i])) / float(prices[i])))) + "\n"
	        	#"    [PTP]  " + str(prev_pos) + "\n"
			#"    ]\n"
	    	 )
		 continue
            position -= 1
	    prior_balance = bank_balance
            bank_balance += (prices[i] - (prices[i] * 0.00075))
	    fees_paid += (prices[i] * 0.00075)
	    trade_count += 1
	    insig = "SIGNIF"
	    if ((abs(float(prior_value) - float(prices[i])) / float(prices[i]))) < 0.00075:
		insig = "INSIG"
	    print(
	        "[synthetic SHORT " + insig + "\n"
	        #"    [ITR]  " + str(i - 720) + "\n"
	        #"    [PPB] $" + str(prior_balance) + "\n"
	        #"    [PTV] $" + str(prior_value) + "\n"
	        #"    [CTV] $" + str(prices[i]) + "\n"
	        #"    [PCD] $" + str(abs(float(prior_value) - float(prices[i]))) + "\n"
	        #"    [PCS]  " + str(((abs(float(prior_value) - float(prices[i])) / float(prices[i])))) + "\n"
	        #"    [PTP]  " + str(prev_pos) + "\n"
	        #"    [CPB] $" + str(bank_balance) + "\n"
		#"    ]\n"
	    )
	    prev_pos = "SHORT"
	    prior_value = prices[i]
	# sell what you bought
    if position == 1:
        bank_balance += prices[len(prices) - 1]
    # pay back what you borrowed
    if position == -1:
        bank_balance -= prices[len(prices) - 1]
    print(
	"[series statistics " + "\n"
	"    [length]  " + str(((i - 720) * 10) / 60) + " min.\n"
	"    [profit] $" + str(bank_balance) + "\n"
	"    [fees paid] $" + str(fees_paid) + "\n"
	"    [revenue] $" + str((fees_paid + bank_balance)) + "\n"
	"    [trades exec.] " + str(trade_count) + "\n"
	"    ]\n"
    )
    return bank_balance
