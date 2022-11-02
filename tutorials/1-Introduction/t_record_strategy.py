#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2022/11/1 18:44
# @File  : t_record_strategy.py
# @Author: 
# @Desc  :
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from data_download import get_daily_stock_bynames
mpl.rcParams['font.family'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
from pyecharts.charts import Bar
from pyecharts.charts import Scatter, Grid
from pyecharts import options as opts
from pyecharts.charts import Line

def t_strategy():
    """
    均价=（收盘价*2+最高价+最低价）/4
    猜测第二天最高价=前一天均价+（前一天最高价-前一天最低价）
    猜测第二天最低价=前一天均价-（前一天最高价-前一天最低价）
    Returns
    -------

    """
    stock_names = ["麒盛科技","奥维通信","三七互娱","宝丰能源","浙江美大"]
    TRAIN_START_DATE = '2010-01-01'
    TRAIN_END_DATE = '2021-05-31'
    data, ts_codes_name = get_daily_stock_bynames(start_date=TRAIN_START_DATE, end_date=TRAIN_END_DATE, stock_names=stock_names)
    print(f"共获取到数据: {data.shape}, 股票名称和代码: {ts_codes_name}")
    data["均价"] =(data["close"] * 2 + data["high"] + data["low"]) /4
    data["猜测最高价"] = data["均价"].shift(-1) + (data["high"].shift(-1) - data["low"].shift(-1))
    data["猜测最低价"] = data["均价"].shift(-1) - (data["high"].shift(-1) - data["low"].shift(-1))
    plot_stock_pyecharts(data, ts_codes_name)

def plot_stock_matplot(data, ts_codes_name, plot_dir="plots"):
    """
    每只股票绘制一个图，横轴是日期，纵轴分别是真实最高价，猜测最高价，真实最低价，猜测最低价，不同颜色表示
    Parameters
    ----------
    data :

    Returns
    -------
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    ts_codes = data["ts_code"].unique()
    name_ts_code = {v:k for k, v in ts_codes_name.items()}
    for code in ts_codes:
        stock_name = name_ts_code[code]
        title = f"股票: {stock_name}, 代码: {code}"
        print(f"绘图: {title}")
        image_name = f"{stock_name}.jpg"
        df = data[data["ts_code"] == code]
        x_date = df["trade_date"].to_list()
        x_date_list = list(range(len(x_date)))
        y_low = df["low"]
        y_high = df["high"]
        y_high_guess = df["猜测最高价"]
        y_low_guess = df["猜测最低价"]
        plt.clf()
        plt.title(title)
        plt.plot(x_date_list, y_low, color="y", label="真实最低价")
        plt.plot(x_date_list, y_high, color="r", label="真实最高价")
        plt.plot(x_date_list, y_high_guess, color="g", label="猜测最高价")
        plt.plot(x_date_list, y_low_guess, color="b", label="猜测最低价")
        plt.legend(loc=2)
        # plt.xlabel(xlabel=x_date)
        # plt.ylabel(ylabel="价格")
        save_png = os.path.join(plot_dir, image_name)
        plt.savefig(save_png)
        print(f"保存绘图到{save_png}文件中")
def plot_stock_pyecharts(data, ts_codes_name, plot_dir="plots"):
    """
    每只股票绘制一个图，横轴是日期，纵轴分别是真实最高价，猜测最高价，真实最低价，猜测最低价，不同颜色表示
    Parameters
    ----------
    data :

    Returns
    -------
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    ts_codes = data["ts_code"].unique()
    name_ts_code = {v:k for k, v in ts_codes_name.items()}
    for code in ts_codes:
        stock_name = name_ts_code[code]
        title = f"股票: {stock_name}, 代码: {code}"
        print(f"绘图: {title}")
        image_name = f"{stock_name}.html"
        html = os.path.join(plot_dir, image_name)
        df = data[data["ts_code"] == code]
        x_date = df["trade_date"].to_list()
        x_day = [i.strftime('%Y%m%d') for i in x_date]
        x_date_list = list(range(len(x_date)))
        y_low = df["low"]
        y_high = df["high"]
        y_high_guess = df["猜测最高价"]
        y_low_guess = df["猜测最低价"]
        line_plot = Line()
        line_plot.add_xaxis(x_day)
        line_plot.add_yaxis("真实最高价", y_high)
        line_plot.add_yaxis("真实最低价", y_low)
        line_plot.add_yaxis("猜测最高价", y_high_guess)
        line_plot.add_yaxis("猜测最低价", y_low_guess)
        line_plot.set_global_opts(title_opts=opts.TitleOpts(title=title))
        line_plot.render(html)
        print(f"保存绘图到{html}文件中")

if __name__ == '__main__':
    t_strategy()