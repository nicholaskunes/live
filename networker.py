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
    depth = requests.get('https://api.pro.coinbase.com/products/BTC-USD/book?level=2').json()
    request_time = time.clock() - start
    date = datetime.now()
    price = float(ticker['side'])
    v_bid = sum([float(bid[1]) for bid in depth['bids']])
    v_ask = sum([float(ask[1]) for ask in depth['asks']])
    #collection.insert({'date': date, 'price': price, 'v_bid': v_bid, 'v_ask': v_ask})
    tickCount += 1;
    print("point: {} req_time: {} date: {} price: {} v_bid: {} v_ask: {}".format(tickCount, request_time, date, price, v_bid, v_ask))


def main():
    """Run tick() at the interval of every ten seconds."""
    scheduler = BlockingScheduler(timezone=utc)
    scheduler.add_job(tick, 'interval', seconds=10)
    scheduler.start()


if __name__ == '__main__':
    main()
