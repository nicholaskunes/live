import requests
import time
from pytz import utc
from datetime import datetime
from pymongo import MongoClient
from apscheduler.schedulers.blocking import BlockingScheduler
import logging

client = MongoClient()
database = client['predictor']
collection = database['gdax']
tickCount = 0;

logging.basicConfig()

def tick():
    global tickCount 
    start = time.clock()
    ticker = requests.get('https://www.bitmex.com/api/v1/orderBook/L2?symbol=XBTUSD&depth=1').json()
    depth = requests.get('https://www.bitmex.com/api/v1/orderBook/L2?symbol=XBTUSD&depth=3').json()
    request_time = time.clock() - start
    date = datetime.now()
    price = float(ticker[0]['price'])
    v_bid = sum([float(bid['size']) for bid in depth if bid['side'] == "Buy"])
    v_ask = sum([float(bid['size']) for bid in depth if bid['side'] == "Sell"])
    collection.insert({'date': date, 'price': price, 'v_bid': v_bid, 'v_ask': v_ask, 'v_bid_inf': float(v_bid / price), 'v_ask_inf': float(v_ask / price)})
    tickCount += 1;
    print("point: {} req_time: {} date: {} price: {} v_bid: {} v_ask: {} v_bid_inf: {} v_ask_inf: {}".format(tickCount, request_time, date, price, v_bid, v_ask, v_bid_inf, v_ask_inf))


def main():
    """Run tick() at the interval of every ten seconds."""
    scheduler = BlockingScheduler(timezone=utc)
    scheduler.add_job(tick, 'interval', seconds=10)
    scheduler.start()


if __name__ == '__main__':
    main()
