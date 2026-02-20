import MetaTrader5 as mt5
import pandas as pd
import time
import numpy as np
from datetime import datetime, timezone

# ================= CONFIGURATION =================
SYMBOL = "XAUUSD"
VOLUME = 0.05
TIMEFRAME_ENTRY = mt5.TIMEFRAME_M5
TIMEFRAME_TREND = mt5.TIMEFRAME_M15
MAGIC_NUMBER = 10101
DEVIATION = 20

# Risk Management
ATR_MULTIPLIER_SL = 1.5  # Stop Loss distance
ATR_MULTIPLIER_TP = 2.0  # Take Profit distance (Conservative 1:1.3 ratio)
MAX_SPREAD_POINTS = 60   # Max spread in Points (Check your broker: usually 60 points = 60 cents on Gold)
RSI_BUY_MIN = 50         # Momentum must be positive
RSI_BUY_MAX = 70         # Don't buy if overbought (Safety)
RSI_SELL_MAX = 50        # Momentum must be negative
RSI_SELL_MIN = 30        # Don't sell if oversold (Safety)

RISK_PERCENT = 1.0          # Risk per trade (1%)
BREAK_EVEN_ATR_MULT = 1.0   # Move SL to BE at 1x ATR
TRAILING_ATR_MULT = 1.0     # Trail SL by 1x ATR
TIME_EXIT_MINUTES = 30      # Close trade after 30 mins if TP/SL not hit

# ================= INITIALIZATION =================
if not mt5.initialize():
    print("MT5 initialization failed. Error:", mt5.last_error())
    quit()

print(f"Bot connected. Trading {SYMBOL}...")

# ================= HELPER FUNCTIONS =================

def get_market_data(symbol, timeframe, count=100):
    """ Fetch data and calculate CORRECT indicators """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    
    if rates is None or len(rates) < count:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # --- 1. EMA Calculation ---
    df['ema_fast'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=50, adjust=False).mean()

    # --- 2. RSI Calculation (Wilder's Smoothing) ---
    change = df['close'].diff()
    gain = change.where(change > 0, 0.0)
    loss = -change.where(change < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # --- 3. ATR Calculation (True Range) ---
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()

    return df

def is_trading_session():
    """ Only trade during high volume hours (London/NY) """
    current_hour = datetime.now(timezone.utc).hour
    # 08:00 UTC to 20:00 UTC (Covers London open to NY Close)
    return 8 <= current_hour <= 20

def get_signal():
    """ 
    Logic: 
    1. Trend (M15) must match Entry (M5).
    2. EMA Cross must have happened.
    3. RSI must be in safe zone (not overbought/oversold).
    4. Uses index -2 (Completed Candle) to prevent repainting.
    """
    df_m5 = get_market_data(SYMBOL, TIMEFRAME_ENTRY)
    df_m15 = get_market_data(SYMBOL, TIMEFRAME_TREND)

    if df_m5 is None or df_m15 is None:
        return "NONE", 0.0

    # Get Completed Candle Data (Index -2)
    # Why -2? Index -1 is the currently moving candle. -2 is the last confirmed one.
    m5_curr = df_m5.iloc[-2]
    m15_curr = df_m15.iloc[-2]
    
    atr_value = m5_curr['atr']

    # --- BUY LOGIC ---
    # 1. M15 Trend is UP
    trend_buy = m15_curr['ema_fast'] > m15_curr['ema_slow']
    # 2. M5 Trend is UP
    entry_buy = m5_curr['ema_fast'] > m5_curr['ema_slow']
    # 3. RSI Safe Zone
    rsi_buy = RSI_BUY_MIN < m5_curr['rsi'] < RSI_BUY_MAX

    if trend_buy and entry_buy and rsi_buy:
        return "BUY", atr_value

    # --- SELL LOGIC ---
    # 1. M15 Trend is DOWN
    trend_sell = m15_curr['ema_fast'] < m15_curr['ema_slow']
    # 2. M5 Trend is DOWN
    entry_sell = m5_curr['ema_fast'] < m5_curr['ema_slow']
    # 3. RSI Safe Zone
    rsi_sell = RSI_SELL_MIN < m5_curr['rsi'] < RSI_SELL_MAX

    if trend_sell and entry_sell and rsi_sell:
        return "SELL", atr_value

    return "NONE", 0.0

def execute_trade(signal, atr):
    # Check Spread
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return
        
    spread = (tick.ask - tick.bid) / mt5.symbol_info(SYMBOL).point
    if spread > MAX_SPREAD_POINTS:
        print(f"Spread too high: {spread} points. Waiting.")
        return

    lot = calculate_lot_size(atr)
    
    # Calculate SL/TP
    if signal == "BUY":
        price = tick.ask
        sl = price - (atr * ATR_MULTIPLIER_SL)
        tp = price + (atr * ATR_MULTIPLIER_TP)
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = tick.bid
        sl = price + (atr * ATR_MULTIPLIER_SL)
        tp = price - (atr * ATR_MULTIPLIER_TP)
        order_type = mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": "Safe Python Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"SUCCESS: {signal} Opened. SL: {sl:.2f}, TP: {tp:.2f}")
    else:
        print(f"FAILED: {result.comment}")

def calculate_lot_size(atr):
    account_info = mt5.account_info()
    symbol_info = mt5.symbol_info(SYMBOL)

    if account_info is None or symbol_info is None:
        return 0.01

    balance = account_info.balance
    risk_amount = balance * (RISK_PERCENT / 100)

    # SL distance in price
    sl_distance = atr * ATR_MULTIPLIER_SL

    # Tick value & contract size
    tick_value = symbol_info.trade_tick_value
    tick_size = symbol_info.trade_tick_size

    if tick_value == 0 or tick_size == 0:
        return 0.01

    lot = risk_amount / (sl_distance / tick_size * tick_value)

    return round(max(lot, symbol_info.volume_min), 2)

def manage_positions():
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return

    symbol_info = mt5.symbol_info(SYMBOL)
    tick = mt5.symbol_info_tick(SYMBOL)

    if symbol_info is None or tick is None:
        return

    point = symbol_info.point

    # Get latest ATR once
    df = get_market_data(SYMBOL, TIMEFRAME_ENTRY)
    if df is None:
        return

    atr = df.iloc[-2]['atr']
    if atr is None or atr == 0:
        return

    for pos in positions:
        ticket = pos.ticket
        price_open = pos.price_open
        sl = pos.sl
        tp = pos.tp
        position_type = pos.type

        current_price = tick.bid if position_type == mt5.ORDER_TYPE_BUY else tick.ask

        profit_distance = abs(current_price - price_open)

        # Small dynamic buffer (10 points)
        buffer = point * 10

        # =========================
        # 1️⃣ BREAK-EVEN (ONLY ONCE)
        # =========================
        break_even_trigger = atr * BREAK_EVEN_ATR_MULT

        if profit_distance >= break_even_trigger:
            if position_type == mt5.ORDER_TYPE_BUY:
                if sl < price_open:
                    new_sl = price_open + buffer
                else:
                    new_sl = sl
            else:
                if sl > price_open:
                    new_sl = price_open - buffer
                else:
                    new_sl = sl

            if new_sl != sl:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": ticket,
                    "sl": new_sl,
                    "tp": tp,
                }
                mt5.order_send(request)
                print(f"Position {ticket} moved to Break-Even")

        # =========================
        # 2️⃣ TRAILING STOP (AFTER BE)
        # =========================
        trail_distance = atr * TRAILING_ATR_MULT

        if position_type == mt5.ORDER_TYPE_BUY:
            if sl >= price_open:  # BE activated
                new_sl = current_price - trail_distance
                if new_sl > sl:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": ticket,
                        "sl": new_sl,
                        "tp": tp,
                    }
                    mt5.order_send(request)
                    print(f"Position {ticket} trailing SL updated")

        else:
            if sl <= price_open:
                new_sl = current_price + trail_distance
                if new_sl < sl:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": ticket,
                        "sl": new_sl,
                        "tp": tp,
                    }
                    mt5.order_send(request)
                    print(f"Position {ticket} trailing SL updated")


# ================= MAIN LOOP =================
print("Starting Strategy Loop...")
while True:
    try:
        # 1. Check Trading Hours
        # if not is_trading_session():
        #     print("Outside trading hours. Sleeping 60s...")
        #     time.sleep(60)
        #     continue

        # 2. Check if we already have a position
        positions = mt5.positions_get(symbol=SYMBOL)
        # if positions and len(positions):
        #     # We are already in a trade, wait for it to hit SL or TP
        #     print(f"Trade active: {len(positions)} position(s). Waiting...")
        #     time.sleep(30)
        #     continue

        # 3. Get Signal
        manage_positions()
        direction, current_atr = get_signal()

        if direction != "NONE":
            print(f"Signal Detected: {direction} (ATR: {current_atr:.2f})")
            execute_trade(direction, current_atr)
            # Sleep to prevent double execution on same candle
            time.sleep(300) 
        else:
            print(f"{datetime.now().time()} - No Signal. Scanning...")
            time.sleep(10)

    except Exception as e:
        print(f"Error in main loop: {e}")
        time.sleep(10)