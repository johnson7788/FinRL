#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2022/8/16 19:54
# @File  : tushare_download.py
# @Author: 
# @Desc  : 测试tushare的下载功能
import os
import requests
import json
import pandas as pd
import tushare as ts
from token_file import TOKEN
import datetime

cache_dir = "/Users/admin/tmp/cache"

def get_today_data(use_cache=True):
    """
    获取今日的数据
    :param use_cache: 是否使用缓存
    :type use_cache: bool
    :return:
    代码，名称，涨跌幅，现价，开盘价，最高价，最低价，今日收盘价，成交量，换手率，成交额，市盈率，市净率，总市值，流通市值
    :rtype:
    """
    cache_file = os.path.join(cache_dir,"today_data.json")
    cache_excel = os.path.join(cache_dir,"today_data.xlsx")
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            df = pd.read_json(f)
    else:
        df = ts.get_today_all()
        df.to_json(cache_file)
        df.to_excel(cache_excel)
    return df

def get_day_stock(stock_code, start, end, use_cache=True):
    """
    获取某只股票的数据, 数据格式
    Index(['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close',
       'change', 'pct_chg', 'vol', 'amount'],
      dtype='object')
    ts_code: 股票代码 例如： 000028.SZ
    trade_date: 交易日期 例如： 20191231
    open: 开盘价 例如： 7.50
    high: 最高价 例如： 7.50
    low: 最低价 例如： 7.50
    close: 收盘价 例如： 7.50
    pre_close: 昨收价 例如： 7.50
    change: 涨跌额 例如： 1.58000
    pct_chg: 涨跌幅 例如： 4.46330
    vol: 成交量 例如： 104316.10000
    amount: 成交额 例如： 384074.24100
    ----------
    stock_code :
    start :
    end :
    use_cache :

    Returns
    -------
    """
    cache_file = os.path.join(cache_dir,f"{stock_code}_{start}_{end}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            df = pd.read_json(f)
    else:
        pro = ts.pro_api(TOKEN)
        # 拉取数据
        df = pro.daily(**{
            "ts_code": stock_code,
            "trade_date": "",
            "start_date": start,
            "end_date": end,
            "offset": "",
            "limit": ""
        }, fields=[
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount"
        ])
        assert df is not None, "获取数据失败"
        df.to_json(cache_file)
    return df

def get_select_stock_data():
    """
    获取训练数据
    :return:
    """
    stock_info = get_api_status(use_cache=True)
    stock_items = stock_info["data"]["items"]
    # 选出20只股票的信息, 过滤名字中不包含"ST"的股票
    no_ST = [item for item in stock_items if "ST" not in item[1]]
    select_stock = no_ST[:20]
    print(f"选出20只股票信息是:{select_stock}")
    return select_stock

def get_train_data(start_date=20100101, end_date=20220101, use_cache=True, format_date=True):
    """
    获取训练数据
    :param start_date: 开始日期, str or int
    :param end_date: 结束日期, str or int
    :param use_cache: 是否使用缓存
    :param format_date: 是否格式化日期为字符串格式
    :return:
    """
    # 如果是字符串格式的日期，变成数字格式的日期
    if isinstance(start_date, str):
        start_date = start_date.replace("-", "")
        start_date = int(start_date)
    if isinstance(end_date, str):
        end_date = end_date.replace("-", "")
        end_date = int(end_date)
    stock_items = get_select_stock_data()
    code_data = {}
    for one_stock in stock_items:
        stock_code = one_stock[0]
        stock_name = one_stock[1]
        one_data = get_day_stock(stock_code, start=start_date, end=end_date, use_cache=use_cache)
        # 把返回的pandas类型的one_data中的trade_date字段变成str类型, 需要format成%Y-%m-%d格式
        if format_date:
            one_data["trade_date"] = one_data["trade_date"].apply(lambda x: str(x))
            one_data["trade_date"] = one_data["trade_date"].apply(lambda x: f"{x[0:4]}-{x[4:6]}-{x[6:8]}")
        code_data[stock_name] = one_data
    print(f"获取训练数据完成，共计{len(code_data)}只股票")
    return code_data

def get_api_status(use_cache=False):
    """
    curl -X POST -d '{"api_name": "stock_basic", "token": "xxxxxxxx", "params": {"list_stauts":"L"}, "fields": "ts_code,name,area,industry,list_date"}' http://api.tushare.pro
    Returns
    -------
    """
    cache_file = os.path.join(cache_dir,"api_status.json")
    if use_cache and os.path.exists(cache_file):
        print(f"注意：使用的是缓存文件{cache_file}")
        with open(cache_file, "r") as f:
            data = json.load(f)
            return data
    url = "http://api.tushare.pro"
    params = {"list_stauts":"L"}
    fields = "ts_code,name,area,industry,list_date"
    data = {"api_name": "stock_basic", "token": TOKEN, "params": params, "fields": fields}
    headers = {'content-type': 'application/json'}
    # 提交form格式数据
    r = requests.post(url, data=json.dumps(data), headers=headers)
    assert r.status_code == 200, "获取数据失败"
    result = r.json()
    print(json.dumps(result, indent=4, ensure_ascii=False))
    with open(cache_file, "w") as f:
        json.dump(result, f, ensure_ascii=False)
        print(f"保存缓存文件{cache_file}")
    return result

if __name__ == '__main__':
    # get_api_status(use_cache=True)
    get_train_data()
