"""Viadrina Fraud Detector: real-time scoring API on IBM LinuxONE.

POST /score     one raw transaction (same fields as transactions_history.csv) -> risk score, tier, reasons
POST /explain   tier + amount + reasons -> one analyst sentence from IBM Granite (fact-checked)
POST /feedback  analyst label for a scored transaction (appended to feedback.jsonl for the next retrain)
GET  /health    status of models and in-memory card history

Features are computed here, per card, with the same definitions as the training notebook, so a client
sends the raw transaction and never pre-computed features.
"""
import bisect
import json
import os
import re
import threading
import time
from collections import deque

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from flask import Flask, jsonify, request

MODEL_DIR = os.environ.get('MODEL_DIR', '/home/ibm-user/shared/models/')
DATA_PATH = os.environ.get('DATA_PATH', '/home/ibm-user/transactions_history.csv')
HISTORY_END = pd.Timestamp(os.environ.get('HISTORY_END', '2016-01-01'))   # card history loaded at start-up
FEEDBACK_PATH = os.environ.get('FEEDBACK_PATH', '/home/ibm-user/shared/feedback.jsonl')
GRANITE_PATH = '/home/ibm-user/granite-4.0-h-350m-base-be.f16.gguf'

with open(os.path.join(MODEL_DIR, 'config.json')) as f:
    CFG = json.load(f)
FEATURE_COLUMNS = CFG['feature_columns']
NN_FEATURES = CFG['nn_features']
W_XGB, W_LSTM, W_TF = CFG['blend_weights']          # order used in the notebook: (XGB, LSTM, TF)
TOP_MCCS = set(CFG['top_mccs'])
SEQ_LEN = CFG['seq_len']
US_STATES = set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO "
                "MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split())

booster = xgb.Booster()
booster.load_model(os.path.join(MODEL_DIR, 'xgb.json'))
tf_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'tf_dense.keras'))
lstm_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'lstm.keras'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
SCALE_MEAN, SCALE_STD = scaler.mean_, scaler.scale_


# ---------- features: same definitions as the notebook, computed one transaction at a time ----------

class CardState:
    """Everything the features need to know about one card's past transactions."""
    def __init__(self):
        self.recent = deque()             # (ts, amount) of the last 7 days
        self.amounts_sorted = []          # all previous amounts, for the running median
        self.last_ts = None
        self.merchants = set()
        self.cities = set()
        self.seq = deque(maxlen=SEQ_LEN - 1)   # scaled NN vectors of the previous transactions (LSTM)


cards = {}
state_lock = threading.Lock()


def parse_amount(value):
    return float(str(value).replace('$', '')) if value is not None else 0.0


def is_missing(value):
    return value is None or (isinstance(value, float) and np.isnan(value))


def build_features(tx):
    """Compute model features for one raw transaction and add it to its card's history."""
    ts = pd.Timestamp(year=int(tx['Year']), month=int(tx['Month']), day=int(tx['Day']),
                      hour=int(str(tx['Time']).split(':')[0]), minute=int(str(tx['Time']).split(':')[1]))
    amount = parse_amount(tx['Amount'])
    city = str(tx['Merchant City']).strip()          # the CSV stores " ONLINE" with a leading space
    state_value = None if is_missing(tx.get('Merchant State')) else tx.get('Merchant State')
    errors = [e.strip() for e in str(tx.get('Errors?') or '').split(',') if e.strip()]

    f = {c: 0.0 for c in FEATURE_COLUMNS}
    f.update(amount=amount, hour=ts.hour, weekday=ts.dayofweek, is_refund=float(amount < 0),
             foreign_merchant=float(state_value is not None and state_value not in US_STATES), MCC=int(tx['MCC']))
    if f'tx_type_{tx["Use Chip"]}' in f:
        f[f'tx_type_{tx["Use Chip"]}'] = 1.0
    for e in errors:
        if f'err_{e}' in f:
            f[f'err_{e}'] = 1.0

    key = (tx['User'], tx['Card'])
    card = cards.setdefault(key, CardState())
    while card.recent and card.recent[0][0] <= ts - pd.Timedelta(days=7):
        card.recent.popleft()
    for name, window in (('count_1h', pd.Timedelta(hours=1)), ('count_24h', pd.Timedelta(hours=24)),
                         ('count_7d', pd.Timedelta(days=7))):
        f[name] = float(sum(1 for t, _ in card.recent if t > ts - window))
    f['amt_24h'] = float(sum(a for t, a in card.recent if t > ts - pd.Timedelta(hours=24)))
    f['mins_since_last'] = (ts - card.last_ts).total_seconds() / 60.0 if card.last_ts is not None else 365 * 24 * 60.0
    n = len(card.amounts_sorted)
    median = 0.0 if n == 0 else (card.amounts_sorted[n // 2] if n % 2 else
                                 (card.amounts_sorted[n // 2 - 1] + card.amounts_sorted[n // 2]) / 2)
    f['ratio_median'] = amount / median if median > 0 else 0.0
    f['first_time_merchant'] = float(tx['Merchant Name'] not in card.merchants)
    f['first_time_city'] = float(city not in card.cities)

    nn = np.array([nn_value(name, f) for name in NN_FEATURES], dtype='float64')
    nn_scaled = np.clip((nn - SCALE_MEAN) / SCALE_STD, -10, 10).astype('float32')
    seq = np.zeros((1, SEQ_LEN, len(NN_FEATURES)), dtype='float32')
    history = list(card.seq) + [nn_scaled]
    seq[0, SEQ_LEN - len(history):] = history

    # add this transaction to the card's history
    card.recent.append((ts, amount))
    bisect.insort(card.amounts_sorted, amount)
    card.last_ts = ts
    card.merchants.add(tx['Merchant Name'])
    card.cities.add(city)
    card.seq.append(nn_scaled)
    return ts, f, nn_scaled, seq, state_value


def nn_value(name, f):
    if name in f:
        return f[name]
    if name.endswith('_log'):
        x = f[name[:-4]]
        return float(np.sign(x) * np.log1p(abs(x)))
    if name.startswith('mcc_'):
        mcc = f['MCC'] if f['MCC'] in TOP_MCCS else -1
        return float(int(name[4:]) == mcc)
    return 0.0


# ---------- reasons: same wording rules as the notebook ----------

MCC_NAMES = {5311: "department store", 5300: "wholesale club", 5310: "discount store", 4829: "money transfer",
             5912: "drug store", 5411: "grocery store", 5651: "family clothing", 5719: "home furnishing",
             5499: "food store", 5812: "restaurant", 5541: "gas station", 3504: "hotel"}


def describe_reason(feat, f, tx, state_value):
    v = f.get(feat)
    if feat in ('amount', 'ratio_median'):
        if f['ratio_median'] >= 1.5:
            return f"Amount ${f['amount']:.2f} is {f['ratio_median']:.1f}x this card's usual amount"
        if 0 < f['ratio_median'] <= 0.3:
            return f"Unusually small amount (${f['amount']:.2f}, {f['ratio_median']:.1f}x usual), typical of card testing"
        return None
    if feat == 'count_1h' and v >= 2:
        return f"{int(v)} other transactions on this card in the last hour"
    if feat == 'count_24h' and v >= 3:
        return f"{int(v)} other transactions on this card in the last 24 hours"
    if feat == 'count_7d' and v >= 10:
        return f"{int(v)} other transactions on this card in the last 7 days"
    if feat == 'amt_24h' and v > 0:
        return f"${v:.2f} already spent on this card in the last 24 hours"
    if feat == 'mins_since_last' and v < 60:
        return f"Only {v:.0f} minutes since this card's previous transaction"
    if feat == 'foreign_merchant' and v == 1:
        return f"Merchant in {state_value}, outside the US"
    if feat == 'first_time_city' and v == 1:
        city = str(tx['Merchant City']).strip()
        return "First purchase at a new location" if city.upper() == 'ONLINE' else f"First purchase ever in {city}"
    if feat == 'first_time_merchant' and v == 1:
        return "First purchase ever at this merchant"
    if feat == 'tx_type_Online Transaction' and v == 1:
        return "Online purchase without the physical card"
    if feat == 'is_refund' and v == 1:
        return f"Refund of ${abs(f['amount']):.2f}"
    if feat == 'MCC':
        rate = MCC_FRAUD_RATE.get(int(v))
        if rate is None or rate <= BASE_FRAUD_RATE:
            return None
        name = MCC_NAMES.get(int(v), "merchant category")
        return f"{name.capitalize()} (MCC {int(v)}): {rate * 100:.2f}% of past purchases were fraud vs {BASE_FRAUD_RATE * 100:.2f}% overall"
    if feat.startswith('err_') and v == 1:
        return f"Error on this attempt: {feat[4:]}"
    return None


def score(tx):
    t0 = time.perf_counter()
    with state_lock:
        ts, f, nn_scaled, seq, state_value = build_features(tx)
    x = np.array([[f[c] for c in FEATURE_COLUMNS]], dtype='float32')
    d = xgb.DMatrix(x, feature_names=FEATURE_COLUMNS)
    p_xgb = float(booster.predict(d)[0])
    contribs = booster.predict(d, pred_contribs=True)[0][:-1]
    # direct model calls: fast for one row, and always use the loaded weights (see the predict() note in the notebook)
    p_tf = float(tf_model(nn_scaled[None, :], training=False).numpy()[0, 0])
    p_lstm = float(lstm_model(seq, training=False).numpy()[0, 0])
    blended = W_XGB * p_xgb + W_LSTM * p_lstm + W_TF * p_tf
    tier = 'RED' if blended >= CFG['red_thr'] else ('YELLOW' if blended >= CFG['yellow_thr'] else 'CLEAR')

    reasons = []
    if tier != 'CLEAR':
        for j in np.argsort(contribs)[::-1]:
            if contribs[j] <= 0 or len(reasons) >= 3:
                break
            text = describe_reason(FEATURE_COLUMNS[j], f, tx, state_value)
            if text and text not in reasons:
                reasons.append(text)
        if len(reasons) < 3 and (p_lstm >= CFG['lstm_thr'] or p_tf >= CFG['tf_thr']):
            reasons.append("Recent sequence of transactions on this card is unlike its history (neural models)")
        reasons = reasons or ["Several weak signals combined; no single dominant factor"]

    return {
        'transaction_id': f"{tx['User']}-{tx['Card']}-{ts:%Y%m%d%H%M}",
        'score': round(blended, 6),
        'action_tier': tier,
        'flagged': tier != 'CLEAR',
        'reasons': reasons,
        'model_scores': {'xgboost': round(p_xgb, 6), 'lstm': round(p_lstm, 6), 'tf_dense': round(p_tf, 6)},
        'latency_ms': round((time.perf_counter() - t0) * 1000, 2),
    }


# ---------- start-up: load card history up to HISTORY_END ----------

def load_history():
    global MCC_FRAUD_RATE, BASE_FRAUD_RATE
    df = pd.read_csv(DATA_PATH)
    df['ts'] = pd.to_datetime(dict(year=df['Year'], month=df['Month'], day=df['Day'],
                                   hour=df['Time'].str.slice(0, 2).astype(int),
                                   minute=df['Time'].str.slice(3, 5).astype(int)))
    df['amount'] = df['Amount'].astype(str).str.replace('$', '', regex=False).astype(float)
    df['is_fraud'] = (df['Is Fraud?'].astype(str).str.strip().str.lower() == 'yes').astype(int)
    train = df[df['Year'] <= 2015]                                # same training period as the notebook
    MCC_FRAUD_RATE = train.groupby('MCC')['is_fraud'].mean().to_dict()
    BASE_FRAUD_RATE = float(train['is_fraud'].mean())
    hist = df[df['ts'] < HISTORY_END].sort_values(['User', 'Card', 'ts', 'amount'])
    for tx in hist.to_dict('records'):
        build_features(tx)
    return len(hist)


HISTORY_ROWS = load_history()


# ---------- Granite (loaded on first use; several seconds per note, so it is a separate call) ----------

llm = None
llm_lock = threading.Lock()
FEW_SHOT = """Rewrite the alert as one sentence for a bank fraud analyst. Use only the facts given.

Alert: RED | Amount: $412.50
Reasons:
- Online purchase without the physical card
- Amount $412.50 is 5.2x this card's usual amount
- 3 other transactions on this card in the last hour
Sentence: RED alert on a $412.50 online purchase, about 5.2 times this card's usual amount, after 3 other transactions on this card in the last hour.

Alert: RED | Amount: refund of $120.00
Reasons:
- Online purchase without the physical card
- Only 4 minutes since this card's previous transaction
- $250.00 already spent on this card in the last 24 hours
Sentence: RED alert on a refund of $120.00 made online, only 4 minutes after this card's previous transaction, with $250.00 already spent on this card in the last 24 hours.

"""
FACT_WORDS = ['pin', 'cvv', 'zip', 'expiration', 'balance', 'glitch', 'card number', 'rome', 'italy', 'mexico',
              'city', 'merchant', 'refund', 'online', 'minute', 'hour', 'outside', 'never', 'new location']


def amount_text(amount):
    return f"refund of ${abs(amount):.2f}" if amount < 0 else f"${amount:.2f}"


def template_note(tier, amount, reasons):
    what = amount_text(amount) if amount < 0 else f"${amount:.2f} transaction"
    return f"{tier} alert on a {what}: " + "; ".join(r[0].lower() + r[1:] for r in reasons) + "."


def granite_note(tier, amount, reasons, max_tokens=80):
    global llm
    with llm_lock:
        if llm is None:
            from llama_cpp import Llama
            llm = Llama(model_path=GRANITE_PATH, n_ctx=1024, n_threads=3, verbose=False)
        prompt = FEW_SHOT + f"Alert: {tier} | Amount: {amount_text(amount)}\nReasons:\n" + \
            "\n".join(f"- {r}" for r in reasons) + "\nSentence:"
        room = llm.n_ctx() - len(llm.tokenize(prompt.encode('utf-8'))) - 8
        budget = min(max_tokens, room)
        for _ in range(2):
            if budget <= 0:
                return None
            out = llm(prompt, max_tokens=budget, stop=["\n", "Alert:"], temperature=0.0, echo=False)
            choice = out['choices'][0]
            text = choice['text'].strip()
            if choice['finish_reason'] == 'stop':
                return text if text.endswith(('.', '!', '?')) else None
            budget = min(budget * 2, room)
        return None


def safe_note(tier, sentence, reasons, amount):
    if sentence:
        facts = (" ".join(reasons) + " " + amount_text(amount)).lower()
        allowed = set(re.findall(r'\d+(?:\.\d+)?', facts))
        found = set(re.findall(r'\d+(?:\.\d+)?', sentence))
        invented = [w for w in FACT_WORDS if w in sentence.lower() and w not in facts]
        if amount < 0 and 'refund' not in sentence.lower():
            invented.append('refund described as a purchase')
        if found <= allowed and not invented:
            return sentence, 'granite'
    return template_note(tier, amount, reasons), 'template'


# ---------- HTTP endpoints ----------

app = Flask(__name__)
REQUIRED = ['User', 'Card', 'Year', 'Month', 'Day', 'Time', 'Amount', 'Use Chip', 'Merchant Name', 'Merchant City', 'MCC']


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models': ['xgboost', 'tf_dense', 'lstm'], 'cards_in_memory': len(cards),
                    'history_rows_loaded': HISTORY_ROWS, 'history_end': str(HISTORY_END.date())})


@app.route('/score', methods=['POST'])
def score_endpoint():
    tx = request.get_json(silent=True)
    if not isinstance(tx, dict):
        return jsonify({'error': 'Send one transaction as a JSON object.'}), 400
    missing = [k for k in REQUIRED if k not in tx]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400
    try:
        return jsonify(score(tx))
    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Could not read the transaction: {e}'}), 400


@app.route('/explain', methods=['POST'])
def explain_endpoint():
    body = request.get_json(silent=True) or {}
    if not body.get('reasons') or 'amount' not in body:
        return jsonify({'error': 'Send tier, amount and reasons (from /score).'}), 400
    t0 = time.perf_counter()
    tier = body.get('tier', 'RED')
    note, source = safe_note(tier, granite_note(tier, float(body['amount']), body['reasons']),
                             body['reasons'], float(body['amount']))
    return jsonify({'note': note, 'source': source, 'latency_ms': round((time.perf_counter() - t0) * 1000, 1)})


@app.route('/feedback', methods=['POST'])
def feedback_endpoint():
    body = request.get_json(silent=True) or {}
    if 'transaction_id' not in body or body.get('label') not in ('fraud', 'not_fraud'):
        return jsonify({'error': "Send transaction_id and label ('fraud' or 'not_fraud')."}), 400
    with open(FEEDBACK_PATH, 'a') as fh:
        fh.write(json.dumps({'transaction_id': body['transaction_id'], 'label': body['label'],
                             'received': pd.Timestamp.now().isoformat()}) + '\n')
    return jsonify({'stored': True})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
