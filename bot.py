# -*- coding: utf-8 -*-
# v12.5.1 - Hibrit PnL Hesaplama (Typo Düzeltmesi ile Tam Sürüm)

import pandas_ta as ta
from binance.client import Client
from binance.enums import *
import pandas as pd
import numpy as np
import time
import logging
import threading
import requests
from typing import Dict, List, Optional, Tuple
import math

# --- API Configuration ---
API_KEY = "N2FZqe4TLxUicEv3YtUoSBHuyo9jfFuKZl9Bg6XHZKvuXz3h3bNvs6msSUQch3JB"      # Buraya kendi API anahtarınızı girin
API_SECRET = "a5MtEUb4Sdv93JMq5uFLke99iy3pHktE5WYwl2gUS0kjqOoyIrRP5TN2phx5mLUU"  # Buraya kendi API gizli anahtarınızı girin

BINANCE_FUTURES_API_BASE_URL = "https://fapi.binance.com"
requests_params = {"timeout": 15}
client = Client(API_KEY, API_SECRET, requests_params=requests_params)

# --- Trading Parameters ---
class TradingConfig:
    DEFAULT_LEVERAGE = 10
    TRADING_TIMEFRAME = "15m"
    HIGHER_TIMEFRAME = "1h"
    EMA_FAST_PERIOD = 12
    EMA_SLOW_PERIOD = 26
    RSI_PERIOD = 14
    ADX_PERIOD = 14
    OBV_MA_PERIOD = 20
    RSI_BULLISH_THRESHOLD = 50
    RSI_BEARISH_THRESHOLD = 50
    ADX_THRESHOLD = 20
    MAX_OPEN_POSITIONS = 10
    MAX_SYMBOLS_TO_MONITOR = 50
    MIN_24H_VOLUME = 5000000
    
    RISK_PER_TRADE_PERCENT = 0.02
    TAKE_PROFIT_USDT = 0.20
    STOP_LOSS_USDT = 0.30
    
    MIN_POSITION_VALUE_USDT = 30.0

config = TradingConfig()

# --- Logging & Global State ---
def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
        fh = logging.FileHandler('trading_bot_v12.5.1.log', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

class TradingState:
    def __init__(self):
        self.wins, self.losses = 0, 0
        self.stop_bot = False
        self.positions: Dict[str, Dict] = {}
        self.symbols: List[str] = []
        self.symbol_info: Dict[str, Dict] = {}
        self.account_balance = 0.0
        self.position_lock = threading.Lock()

configure_logging()
state = TradingState()

# --- Helper, Data & Indicator Functions ---
def calculate_engulfing(df: pd.DataFrame) -> pd.Series:
    prev_open, prev_close = df['open'].shift(1), df['close'].shift(1)
    bullish = (prev_close < prev_open) & (df['close'] > df['open']) & (df['close'] > prev_open) & (df['open'] < prev_close)
    bearish = (prev_close > prev_open) & (df['close'] < df['open']) & (df['close'] < prev_open) & (df['open'] > prev_close)
    return np.where(bullish, 100, np.where(bearish, -100, 0))

def get_symbol_precision():
    try:
        info = client.futures_exchange_info()
        for s in info['symbols']:
            if s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT':
                filters = {f['filterType']: f for f in s['filters']}
                if 'PRICE_FILTER' in filters and 'LOT_SIZE' in filters and 'MIN_NOTIONAL' in filters:
                    state.symbol_info[s['symbol']] = {'price_tick_size': float(filters['PRICE_FILTER']['tickSize']),'quantity_step_size': float(filters['LOT_SIZE']['stepSize']),'min_notional': float(filters['MIN_NOTIONAL'].get('notional', 0.0))}
    except Exception as e: logging.error(f"Error fetching symbol precision: {e}")

def get_ohlcv(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        if not klines: return pd.DataFrame()
        df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote", "ignore"])
        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        logging.warning(f"[{symbol}] Could not fetch OHLCV for {interval}: {e}")
        return pd.DataFrame()

def get_usdt_balance() -> float:
    try:
        balances = client.futures_account_balance()
        return next((float(b['balance']) for b in balances if b['asset'] == 'USDT'), 0.0)
    except Exception as e:
        logging.error(f"Could not fetch USDT balance: {e}")
        return 0.0

def add_indicators(df: pd.DataFrame):
    if df.empty: return df
    df.ta.ema(length=config.EMA_FAST_PERIOD, append=True)
    df.ta.ema(length=config.EMA_SLOW_PERIOD, append=True)
    df.ta.rsi(length=config.RSI_PERIOD, append=True)
    df.ta.adx(length=config.ADX_PERIOD, append=True)
    df.ta.obv(append=True)
    df[f'OBV_MA_{config.OBV_MA_PERIOD}'] = df['OBV'].rolling(window=config.OBV_MA_PERIOD).mean()
    df['CDL_ENGULFING'] = calculate_engulfing(df)
    return df

def refresh_symbols() -> List[str]:
    logging.info("Refreshing symbol list based on 24h volume...")
    fallback_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    try:
        valid_symbols = {s['symbol'] for s in client.futures_exchange_info()['symbols'] if s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'}
        tickers = requests.get(f"{BINANCE_FUTURES_API_BASE_URL}/fapi/v1/ticker/24hr", timeout=10).json()
        volume_data = [{'symbol': t['symbol'], 'volume': float(t['quoteVolume'])} for t in tickers if t.get('symbol') in valid_symbols and float(t.get('quoteVolume', 0)) >= config.MIN_24H_VOLUME]
        sorted_symbols_data = sorted(volume_data, key=lambda x: x['volume'], reverse=True)
        top_symbols = [item['symbol'] for item in sorted_symbols_data[:config.MAX_SYMBOLS_TO_MONITOR]]
        if not top_symbols: return fallback_symbols
        logging.info(f"Monitoring top {len(top_symbols)} symbols. Top 5: {', '.join(top_symbols[:5])}...")
        return top_symbols
    except Exception as e:
        logging.error(f"Error while refreshing symbols: {e}")
        return fallback_symbols

def reconcile_open_positions():
    try:
        all_positions = client.futures_position_information()
        real_open_symbols = {p['symbol'] for p in all_positions if float(p.get('positionAmt', '0')) != 0}
        with state.position_lock:
            bot_open_symbols = set(state.positions.keys())
            closed_externally = bot_open_symbols - real_open_symbols
            if closed_externally:
                for symbol in closed_externally:
                    logging.warning(f"[{symbol}] Position mismatch! Binance'te kapalı ama botta açık görünüyor. Bot state'inden kaldırılıyor.")
                    del state.positions[symbol]
            new_external_positions = real_open_symbols - bot_open_symbols
            if new_external_positions:
                for p in all_positions:
                    if p['symbol'] in new_external_positions:
                        logging.warning(f"[{p['symbol']}] External position detected! Binance'te açık ama botta kayıtlı değil. Bota ekleniyor.")
                        load_single_position(p)
    except Exception as e: logging.error(f"Error during position reconciliation: {e}")

def load_single_position(p: dict):
    position_amt = float(p.get('positionAmt', '0'))
    symbol = p.get('symbol')
    if position_amt != 0 and symbol:
        entry_price = float(p.get('entryPrice', '0'))
        direction = "LONG" if position_amt > 0 else "SHORT"
        quantity = abs(position_amt)
        with state.position_lock:
            state.positions[symbol] = {'direction': direction, 'entry_price': entry_price, 'quantity': quantity}
        logging.info(f"[{symbol}] PRE-EXISTING/EXTERNAL {direction} position monitoring started. Qty: {quantity}, Entry: {entry_price}")
        threading.Thread(target=monitor_position, args=(symbol, entry_price, direction, quantity), daemon=True).start()

def load_existing_positions():
    logging.info("Checking for any existing open positions to monitor...")
    try:
        all_positions = client.futures_position_information()
        for p in all_positions:
            if float(p.get('positionAmt', '0')) != 0:
                load_single_position(p)
    except Exception as e: logging.error(f"Could not load existing positions. Error: {e}")

# --- Trading Logic & Position Management ---
def get_signal(symbol: str, df_trade: pd.DataFrame, df_htf: pd.DataFrame) -> Optional[str]:
    if df_htf.empty or df_trade.empty: return None
    last_htf = df_htf.iloc[-1]
    htf_ema_slow = last_htf.get(f'EMA_{config.EMA_SLOW_PERIOD}')
    if htf_ema_slow is None or pd.isna(htf_ema_slow): return None
    allowed_direction = "LONG" if last_htf['close'] > htf_ema_slow else "SHORT"
    last_trade = df_trade.iloc[-1]
    adx, dmp, dmn = last_trade.get(f'ADX_{config.ADX_PERIOD}'), last_trade.get(f'DMP_{config.ADX_PERIOD}'), last_trade.get(f'DMN_{config.ADX_PERIOD}')
    if any(v is None or pd.isna(v) for v in [adx, dmp, dmn]) or adx < config.ADX_THRESHOLD: return None
    
    ### HATA BURADAYDI, DÜZELTİLDİ ###
    ema_fast, ema_slow, rsi = last_trade.get(f'EMA_{config.EMA_FAST_PERIOD}'), last_trade.get(f'EMA_{config.EMA_SLOW_PERIOD}'), last_trade.get(f'RSI_{config.RSI_PERIOD}')
    
    if any(v is None or pd.isna(v) for v in [ema_fast, ema_slow, rsi]): return None
    base_signal = None
    if ema_fast > ema_slow and rsi > config.RSI_BULLISH_THRESHOLD and dmp > dmn: base_signal = "LONG"
    elif ema_fast < ema_slow and rsi < config.RSI_BEARISH_THRESHOLD and dmn > dmp: base_signal = "SHORT"
    if base_signal != allowed_direction: return None
    obv, obv_ma = last_trade.get('OBV'), last_trade.get(f'OBV_MA_{config.OBV_MA_PERIOD}')
    if any(v is None or pd.isna(v) for v in [obv, obv_ma]) or not ((base_signal == "LONG" and obv > obv_ma) or (base_signal == "SHORT" and obv < obv_ma)): return None
    engulfing_signal = last_trade.get('CDL_ENGULFING')
    is_engulfing_confirmed = (base_signal == "LONG" and engulfing_signal == 100) or (base_signal == "SHORT" and engulfing_signal == -100)
    if not is_engulfing_confirmed: return None
    logging.info(f"[{symbol}] Signal Check Complete | >>> Found fully confirmed {base_signal} signal. <<<")
    return base_signal

def execute_trade(symbol: str, direction: str, quantity: float):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=config.DEFAULT_LEVERAGE)
        logging.info(f"[{symbol}] Leverage set to {config.DEFAULT_LEVERAGE}x")
        order = client.futures_create_order(symbol=symbol, side=SIDE_BUY if direction == "LONG" else SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=quantity)
        logging.info(f"EXECUTED {direction} order for {symbol}, Qty: {quantity}")
        return order
    except Exception as e:
        logging.error(f"Order execution failed for {symbol}: {e}")
        return None

def close_position(symbol: str):
    try:
        with state.position_lock:
            if symbol not in state.positions: return False
        pos_info = client.futures_position_information(symbol=symbol)[0]
        pos_amt = float(pos_info.get('positionAmt', '0'))
        if pos_amt != 0:
            qty_to_close = abs(pos_amt)
            side_to_close = SIDE_SELL if pos_amt > 0 else SIDE_BUY
            client.futures_create_order(symbol=symbol, side=side_to_close, type=ORDER_TYPE_MARKET, quantity=qty_to_close, reduceOnly=True)
            logging.info(f"CLOSED position for {symbol} by sending a reduce-only order.")
            return True
        else: return True
    except Exception as e:
        logging.error(f"Failed to close position for {symbol}: {e}")
    return False

def monitor_position(symbol: str, entry_price: float, direction: str, quantity: float):
    logging.info(f"[{symbol}] Monitoring started. Qty: {quantity}, Entry: {entry_price:.5f}, TP: +${config.TAKE_PROFIT_USDT}, SL: -${config.STOP_LOSS_USDT}")
    is_position_active = True
    while is_position_active and not state.stop_bot:
        try:
            with state.position_lock:
                if symbol not in state.positions:
                    is_position_active = False
                    continue
            pos_info_list = client.futures_position_information(symbol=symbol)
            if not pos_info_list or float(pos_info_list[0].get('positionAmt', '0')) == 0:
                break
            current_mark_price = float(client.futures_mark_price(symbol=symbol)['markPrice'])
            if direction == "LONG":
                unrealized_pnl_usdt = (current_mark_price - entry_price) * quantity
            else:
                unrealized_pnl_usdt = (entry_price - current_mark_price) * quantity
            api_pnl = float(pos_info_list[0].get('unrealizedProfit', '0.0'))
            logging.info(f"[{symbol}] MONITOR | PNL (Calc): ${unrealized_pnl_usdt:.4f} (API: ${api_pnl:.4f}) | TP: +${config.TAKE_PROFIT_USDT:.2f} | SL: -${config.STOP_LOSS_USDT:.2f}")
            if unrealized_pnl_usdt >= config.TAKE_PROFIT_USDT:
                logging.info(f"[{symbol}] FIXED PROFIT TARGET HIT! PnL: ${unrealized_pnl_usdt:.4f}. Closing position.")
                if close_position(symbol): state.wins += 1
                is_position_active = False
            elif unrealized_pnl_usdt <= -config.STOP_LOSS_USDT:
                logging.warning(f"[{symbol}] FIXED STOP-LOSS HIT! PnL: ${unrealized_pnl_usdt:.4f}. Closing position.")
                if close_position(symbol): state.losses += 1
                is_position_active = False
            time.sleep(5)
        except Exception as e:
            logging.error(f"Error in monitor_position for {symbol}: {e}")
            is_position_active = False
    with state.position_lock:
        if symbol in state.positions:
            del state.positions[symbol]
            logging.info(f"[{symbol}] Position removed from bot's internal state.")

# --- Main Trading Loop ---
def run_strategy():
    logging.info(f"Advanced Trading Bot v12.5.1 Started (Typo Fix).")
    state.account_balance = get_usdt_balance()
    if state.account_balance <= 0:
        logging.critical("Could not get a valid account balance. Bot is shutting down.")
        return
    logging.info(f"Initial USDT Balance: {state.account_balance:.2f}")
    get_symbol_precision()
    load_existing_positions()
    state.symbols = refresh_symbols()
    if not state.symbols: logging.critical("Symbol list is empty. Cannot start trading cycle.")
    last_symbol_refresh_time = time.time()

    while not state.stop_bot:
        try:
            reconcile_open_positions()
            if time.time() - last_symbol_refresh_time > 3600:
                state.account_balance = get_usdt_balance()
                logging.info(f"Hourly refresh. Current USDT Balance: {state.account_balance:.2f}")
                new_symbols = refresh_symbols()
                if new_symbols: state.symbols = new_symbols
                last_symbol_refresh_time = time.time()
            
            with state.position_lock:
                open_position_count = len(state.positions)
                current_open_positions = list(state.positions.keys())

            if open_position_count >= config.MAX_OPEN_POSITIONS:
                logging.info(f"Max positions ({open_position_count}/{config.MAX_OPEN_POSITIONS}) reached. Monitoring: {current_open_positions}")
                time.sleep(20)
                continue
            
            logging.info(f"Scanning for new trades... Open Positions: {open_position_count}/{config.MAX_OPEN_POSITIONS}")
            for symbol in state.symbols:
                if state.stop_bot: break
                
                with state.position_lock:
                    is_position_open = symbol in state.positions
                if is_position_open: continue
                
                df_15m, df_1h = get_ohlcv(symbol, config.TRADING_TIMEFRAME), get_ohlcv(symbol, config.HIGHER_TIMEFRAME)
                if df_15m.empty or df_1h.empty: continue
                df_15m, df_1h = add_indicators(df_15m), add_indicators(df_1h)
                
                signal = get_signal(symbol, df_15m, df_1h)
                if signal:
                    s_info = state.symbol_info.get(symbol)
                    if not s_info: continue
                    entry_price = float(client.futures_mark_price(symbol=symbol)['markPrice'])
                    
                    if config.STOP_LOSS_USDT <= 0: continue

                    ideal_risk_usdt = state.account_balance * config.RISK_PER_TRADE_PERCENT
                    quantity = ideal_risk_usdt / config.STOP_LOSS_USDT
                    position_value = quantity * entry_price
                    
                    min_notional_from_exchange = s_info['min_notional']
                    effective_min_notional = max(config.MIN_POSITION_VALUE_USDT, min_notional_from_exchange)

                    if position_value < effective_min_notional:
                        logging.warning(f"[{symbol}] Risk-based position value (${position_value:.2f}) is below minimum (${effective_min_notional:.2f}). Adjusting size to minimum.")
                        quantity = (effective_min_notional / entry_price) * 1.01 
                    
                    margin_required = (quantity * entry_price) / config.DEFAULT_LEVERAGE
                    if margin_required > state.account_balance * 0.5: 
                        logging.error(f"[{symbol}] SAFETY BRAKE! Adjusted position requires too much margin ({margin_required:.2f} USDT). Trade aborted.")
                        continue

                    step_size = s_info['quantity_step_size']
                    quantity = math.floor(quantity / step_size) * step_size
                    
                    if quantity <= 0:
                        logging.warning(f"[{symbol}] Final quantity is zero or less after calculations. Skipping trade.")
                        continue

                    final_position_value = quantity * entry_price
                    logging.info(f"[{symbol}] All checks passed! Attempting to execute {signal} trade.")
                    logging.info(f"[{symbol}] Entry: {entry_price:.5f}, Qty: {quantity}, Position Value: ~${final_position_value:.2f}")
                    
                    order = execute_trade(symbol, signal, quantity)
                    if order:
                        with state.position_lock:
                            state.positions[symbol] = {'direction': signal, 'entry_price': entry_price, 'quantity': quantity}
                        threading.Thread(target=monitor_position, args=(symbol, entry_price, signal, quantity), daemon=True).start()
                
                time.sleep(2) 

            with state.position_lock:
                open_position_count = len(state.positions)
            logging.info(f"--- Cycle finished. Open Positions: {open_position_count} | Wins: {state.wins}, Losses: {state.losses} ---")
            time.sleep(30)
            
        except Exception as e:
            logging.error(f"Main loop critical error: {e}", exc_info=True)
            time.sleep(60)

# --- Program Entry Point ---
if __name__ == "__main__":
    try:
        run_strategy()
    except KeyboardInterrupt:
        state.stop_bot = True
        logging.info("Bot stopping... Closing all positions...")
        with state.position_lock:
            positions_to_close = list(state.positions.keys())
        for sym in positions_to_close:
            close_position(sym)
        logging.info("Waiting for monitoring threads to finish...")
        time.sleep(10)
        logging.info("Bot shut down.")