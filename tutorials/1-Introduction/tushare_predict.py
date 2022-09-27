#!/usr/bin/env python
# coding: utf-8
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import time
import pymysql
import argparse
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.data_processor import DataProcessor

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint
import itertools
from finrl import config
from finrl import config_tickers
import os
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

logfile = "predict.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(logfile, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

from data_download import get_daily_stock_and_indicator, INDICATORS

def prepare_dir():
    # 创建目录
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

def download_data(TRAIN_START_DATE, TRADE_END_DATE, mini=False):
    stock_data = get_daily_stock_and_indicator(TRAIN_START_DATE, TRADE_END_DATE, mini=mini)
    # 变成pandas的dataframe, 新加一个空的pandas，然后把所有数据添加进去
    # 更改列名, ts_code --> tic, vol --> volume, trade_date ==>date
    stock_data.rename(columns={"ts_code": "tic", "vol": "volume", "trade_date": "date"}, inplace=True)
    # 列名: Index(['date', 'open', 'high', 'low', 'close', 'volume', 'tic', 'day'], dtype='object')
    print(f"数据示例：")
    print(stock_data.sort_values(['date', 'tic'], ignore_index=True).head())
    return stock_data

def preprocess_data(df):
    # Step4: 处理数据集
    # 处理丢失数据，和特征工程，吧数据点变成状态
    # * **添加技术指标**。在实际交易中，需要考虑到各种信息，如历史价格、当前持有的股票、技术指标等。在此，我们演示两个趋势跟踪的技术指标。MACD和RSI。
    # * **增加动荡指数**。风险规避反映了投资者是否倾向于保护资本。它也影响了一个人在面对不同市场波动水平时的交易策略。为了控制最坏情况下的风险，如2007-2008年的金融危机，FinRL采用了衡量资产价格极端波动的动荡指数。
    print(f"开始处理数据集, 数据集的形状是: {df.shape}")
    processed_full = df.sort_values(['date','tic'])

    #Step 5: 数据分割和在OpenAI Gym风格中建立一个市场环境
    # 训练过程包括观察股票价格变化，采取动作和奖励的计算。通过与市场环境的互动，agent最终会得出一个可能使（预期）回报最大化的交易策略。
    # # 我们的市场环境，基于OpenAI Gym，用历史市场数据模拟股票市场。

    # 我们将数据分为训练集和测试集，如下所示。
    # 训练数据: 2009-01-01 to 2020-07-01
    #
    # 测试数据: 2020-07-01 to 2021-10-31
    #

    train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)
    trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)
    assert train.empty == False, "训练集是空的"
    assert trade.empty == False, "交易测试集是空的"
    print(f"训练数据集大小：{len(train)}")
    print(f"测试数据集大小：{len(trade)}")

    print(f"部分训练数据集", train.tail())
    print(f"部分测试数据集", trade.head())
    print(f"交易指标：", INDICATORS)
    return train, trade, processed_full

def setup_env(train):
    """
    准备训练环境
    Parameters
    ----------
    train :
    trade :

    Returns
    -------

    """
    stock_dimension = len(train.tic.unique())  #股票数量
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    print(f"股票数量: {stock_dimension}, 状态空间: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension  #默认每只股票的数量

    env_kwargs = {
        "hmax": 2000,  #1000表示，买10手股票
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "make_plots": True,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    # 初始化训练环境
    e_train_gym = StockTradingEnv(df = train, **env_kwargs)

    # 训练环境
    #
    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))
    return env_train, env_kwargs


# Step6: 训练DRL Agent
# * DRL算法来自**Stable Baselines 3**。我们也鼓励用户尝试**ElegantRL**和**Ray RLlib**。
# * FinRL包括微调的标准DRL算法，如DQN、DDPG、多agentDDPG、PPO、SAC、A2C和TD3。我们还允许用户通过改编这些DRL算法来设计自己的DRL算法。

# ### 包括： 5 algorithms (A2C, DDPG, PPO, TD3, SAC)

# ### Agent 1: A2C
def a2c(env_train, total_timesteps):
    agent = DRLAgent(env=env_train)
    model_a2c = agent.get_model("a2c")

    trained_a2c = agent.train_model(model=model_a2c,
                                    tb_log_name='a2c',
                                    total_timesteps=total_timesteps)
    return trained_a2c


# ### Agent 2: DDPG


def ddpg(env_train, total_timesteps):
    agent = DRLAgent(env=env_train)
    model_ddpg = agent.get_model("ddpg")

    trained_ddpg = agent.train_model(model=model_ddpg,
                                     tb_log_name='ddpg',
                                     total_timesteps=total_timesteps)
    return trained_ddpg


# ### Agent 3: PPO


def ppo(env_train, total_timesteps):
    agent = DRLAgent(env=env_train)
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,
    }
    model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

    trained_ppo = agent.train_model(model=model_ppo,
                                    tb_log_name='ppo',
                                    total_timesteps=total_timesteps)
    return trained_ppo


# ### Agent 4: TD3

def td3(env_train, total_timesteps):
    agent = DRLAgent(env = env_train)
    TD3_PARAMS = {"batch_size": 100,
                  "buffer_size": 1000000,
                  "learning_rate": 0.001}

    model_td3 = agent.get_model("td3",model_kwargs = TD3_PARAMS)


    trained_td3 = agent.train_model(model=model_td3,
                                 tb_log_name='td3',
                                 total_timesteps=total_timesteps)
    return trained_td3


def sac(env_train, total_timesteps=30000):
    # ### Agent 5: SAC
    agent = DRLAgent(env=env_train)
    SAC_PARAMS = {
        "batch_size": 128,
        "buffer_size": 1000000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    }

    model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

    trained_sac = agent.train_model(model=model_sac,
                                    tb_log_name='sac',
                                    total_timesteps=total_timesteps)
    return trained_sac


def trade_test_data(trained_model, trade, processed_full, env_kwargs):
    # 假设初始资本为1,000,000美元。

    # ### Set turbulence threshold
    # 设置湍流阈值，使之大于样本内湍流数据的最大值。如果当前的湍流指数大于阈值，那么我们就认为当前的市场是波动的

    data_risk_indicator = processed_full[
        (processed_full.date < TRAIN_END_DATE) & (processed_full.date >= TRAIN_START_DATE)]
    insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=['date'])

    # ### Trading (Out-of-sample Performance)
    #
    # 我们定期更新，以便充分利用数据，例如，每季度、每月或每周重新训练。我们还沿途调整参数，在这个笔记本中，我们使用2009-01到2020-07的样本内数据来调整一次参数，所以随着交易日期长度的延长，这里有一些α衰减。
    #
    # 众多的超参数--例如学习率、训练的样本总数--影响学习过程，通常通过测试一些变化来确定。

    e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)
    # env_trade, obs_trade = e_trade_gym.get_sb_env()

    print(f"测试数据:", trade.head())
    # df_account_value: 每条天的资产价值
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_model,
        environment=e_trade_gym)
    print(df_account_value.shape)
    print(df_account_value.tail())
    print(df_actions.head())
    return df_account_value, df_actions

def backtest(df_account_value, result_file):
    # # Step7: 回测结果
    # 回溯测试在评估交易策略的表现方面起着关键作用。自动回测工具是首选，因为它减少了人为错误。我们通常使用Quantopian pyfolio软件包来回测我们的交易策略。它很容易使用，由各种单独的图组成，提供了交易策略表现的全面图像。

    # ## 7.1 BackTestStats
    # 传入df_account_value，该信息存储在env类中。

    print("==============获取回测结果===========")

    perf_stats_all = backtest_stats(account_value=df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_excel(result_file, index=False)
    print(f"保存回测结果到: {result_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="强化学习预测")
    parser.add_argument('-m', '--model', type=str, default="sac",choices=("sac","ppo","a2c","ddpg","td3","all") ,help='使用哪个模型进行训练')
    parser.add_argument('-st', '--start_train', default='2010-01-01', help='训练的开始时间')
    parser.add_argument('-et', '--end_train', default='2020-05-31', help='训练的结束时间')
    parser.add_argument('-se', '--start_test', default='2020-06-01', help='测试的开始时间')
    parser.add_argument('-ee', '--end_test', default='2022-05-31', help='测试的结束时间')
    parser.add_argument('-t', '--timesteps', type=int, default=200000, help='训练的时间步')
    parser.add_argument('-mi', '--mini', action='store_true', help='迷你数据集')
    args = parser.parse_args()
    # Step3 下载数据集
    TRAIN_START_DATE = args.start_train
    TRAIN_END_DATE = args.end_train
    TRADE_START_DATE = args.start_test
    TRADE_END_DATE = args.end_test
    model = args.model
    prepare_dir()
    print(f"训练日期是: {TRAIN_START_DATE} 到 {TRAIN_END_DATE}, 预测日期是: {TRADE_START_DATE} 到 {TRADE_END_DATE}")
    print(f"使用的模型是: {model}")
    data_train = download_data(TRAIN_START_DATE, TRADE_END_DATE, mini=args.mini)
    train_data, trade_data, processed_full = preprocess_data(data_train)
    env_train, env_kwargs = setup_env(train_data)
    if model == "sac":
        trained_model = sac(env_train, total_timesteps=args.timesteps)
    elif model =="td3":
        trained_model = td3(env_train, total_timesteps=args.timesteps)
    elif model =="ppo":
        trained_model = ppo(env_train, total_timesteps=args.timesteps)
    elif model =="a2c":
        trained_model = a2c(env_train, total_timesteps=args.timesteps)
    elif model =="ddpg":
        trained_model = ddpg(env_train, total_timesteps=args.timesteps)
    elif model == "all":
        for model_name in ["sac","td3","ppo","a2c","ddpg"]:
            print(f"使用的模型是: {model_name}")
            model_func = eval(model_name)
            trained_model = model_func(env_train, total_timesteps=args.timesteps)
            df_account_value, df_actions = trade_test_data(trained_model=trained_model, trade=trade_data,
                                               processed_full=processed_full, env_kwargs=env_kwargs)
            now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
            trade_action_file = f"action_{model_name}_{now}.xlsx"
            df_actions.to_excel(trade_action_file, index=False)
            # 缓存df_account_value到本地
            df_account_value_pkl_file = "cache/df_account_value.pkl"
            df_account_value.to_pickle(df_account_value_pkl_file)
            csv_file = f"backtest_{model_name}_{now}.xlsx"
            backtest(df_account_value, result_file=csv_file)
        print(f"结束所有模型的训练学习")
        sys.exit(0)
    else:
        print(f"不支持的模型,退出")
    df_account_value, df_actions = trade_test_data(trained_model=trained_model, trade=trade_data, processed_full=processed_full, env_kwargs=env_kwargs)
    # 缓存df_account_value到本地
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    trade_action_file = f"action_{model}_{now}.xlsx"
    df_actions.to_excel(trade_action_file, index=False)
    df_account_value_pkl_file = "cache/df_account_value.pkl"
    df_account_value.to_pickle(df_account_value_pkl_file)
    csv_file = f"backtest_{model}_{now}.xlsx"
    backtest(df_account_value, result_file=csv_file)