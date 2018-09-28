# -*- coding: utf-8 -*-
from pymongo import MongoClient
from bayesian_regression import *
import subprocess
import time
import requests
from datetime import datetime
from tqdm import tqdm

client = MongoClient()
database = client['predictor']
collection = database['gdax']

positive = 0

while True:
	prices = []
	v_ask = []
	v_bid = []
	num_points = 777600
	for doc in collection.find().limit(num_points):
		prices.append(doc['price'])
		v_ask.append(doc['v_ask'])
		v_bid.append(doc['v_bid'])

	[tprices1, tprices2] = np.array_split(prices, 2)
	[tv_bid1, tv_bid2] = np.array_split(v_bid, 2)
	[tv_ask1, tv_ask2] = np.array_split(v_ask, 2)
	
	timeseries180 = generate_timeseries(tprices1, 180)
	timeseries360 = generate_timeseries(tprices1, 360)
	timeseries720 = generate_timeseries(tprices1, 720)
	
	centers180 = find_cluster_centers(timeseries180, 100)
	s1 = choose_effective_centers(centers180, 20)

	centers360 = find_cluster_centers(timeseries360, 100)
	s2 = choose_effective_centers(centers360, 20)

	centers720 = find_cluster_centers(timeseries720, 100)
	s3 = choose_effective_centers(centers720, 20)

	Dpi_r, Dp = linear_regression_vars(tprices2, tv_bid2, tv_ask2, s1, s2, s3)

	w = find_parameters_w(Dpi_r, Dp)

	prices4 = []
	iterator = 0
	completion = 0
    	position = 0
	balance = 0
	#for i in tqdm(range(0, 720, 1)): 
	for i in range(0, 720, 1): 
		completion += 1
		prices = []
		v_ask = []
		v_bid = []
		num_points = 777600
		for doc in collection.find().limit(num_points):
			prices.append(doc['price'])
			v_ask.append(doc['v_ask'])
			v_bid.append(doc['v_bid'])
			
		[prices1, prices2] = np.array_split(prices, 2)
		[v_bid1, v_bid2] = np.array_split(v_bid, 2)
		[v_ask1, v_ask2] = np.array_split(v_ask, 2)
	
		[pfluke1, pfluke2, pfluke3] = np.array_split(prices, 3)
		[bfluke1, bfluke2, bfluke3] = np.array_split(v_bid, 3)
		[afluke1, afluke2, afluke3] = np.array_split(v_ask, 3)

	        end = predict(prices2, v_bid2, v_ask2, s1, s2, s3, w)
		endf = predict_flawed(pfluke3, bfluke3, afluke3, s1, s2, s3, w)
		
		ticker = requests.get('https://api.gdax.com/products/BTC-USD/ticker').json()
		curprice = float(ticker['price'])
		prices4.append(curprice)
		
		print "time: " + str(datetime.now()) + "change_variables = [ price: " + str(curprice) + " Δp " + str(end) + " Δp_f " + str(endf) + " ]"
		
        	# BUY
    		if end > 0.15 and position <= 0:
			iterator += 1
    			position += 1
    		        balance -= curprice
			print "[" + str(iterator) + " BUY] " + str(datetime.now()) + " predict t+10s Δp " + str(end) + " $" + str(round(balance, 5))
        	# SELL
    		if end < -0.15 and position >= 0:
			iterator += 1
    			position -= 1
    			balance += curprice
			print "[" + str(iterator) + " SELL] " + str(datetime.now()) + " predict t+10s Δp " + str(end) + " $" + str(round(balance, 5))
		time.sleep(15)
		
	ticker = requests.get('https://api.gdax.com/products/BTC-USD/ticker').json()

	# SELL
    	if position == 1:
        	balance += float(ticker['price'])
    	# PAY BROKER BACK
    	if position == -1:
		balance -= float(ticker['price'])
		
	print "[series profit: $" + str(balance) + " ] " + "trade count: " + str(iterator)
		
	#np.savetxt("btc.csv", dps, delimiter=",")
 	np.savetxt("prices.csv", prices4, delimiter=",")

 	#output = subprocess.check_output("curl --upload-file ./btc.csv https://transfer.sh/btc.csv", shell=True)
  	#output2 = subprocess.check_output("curl --upload-file ./prices.csv https://transfer.sh/prices.csv", shell=True)

 	#subprocess.call("rm btc.csv", shell=True)
	#subprocess.call("rm prices.csv", shell=True)
