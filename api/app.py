import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from flask import Flask, request, jsonify
from llama_cpp import Llama

app = Flask(__name__)

M = '/home/ibm-user/shared/models/'
config = json.load(open(os.path.join(M, 'config.json')))
FEATURE_COLUMNS = config['feature_columns']
NN_FEATURES = config['nn_features']
best_w = config['blend_weights']
red_thr = config['red_thr']
yellow_thr = config['yellow_thr']

xgb_model = xgb.Booster()
xgb_model.load_model(os.path.join(M, 'xgb.json'))
tf_model = tf.keras.models.load_model(os.path.join(M, 'tf_dense.keras'))
scaler = joblib.load(os.path.join(M, 'scaler.joblib'))

def describe_reason(feat, row):
    v = row.get(feat, 0)
    if feat == 'amount' or feat == 'ratio_median':
        if row.get('ratio_median', 1.0) >= 1.5:
            return f"Amount ${row.get('amount', 0):.2f} is {row.get('ratio_median', 1.0):.1f}x this card's usual amount"
        return None
    if feat == 'count_1h' and v >= 2:
        return f"{int(v)} other transactions on this card in the last hour"
    return None

def generate_reasons(xgb_contribs, row):
    contribs = xgb_contribs[:-1]
    reasons = []
    for j in np.argsort(contribs)[::-1]:
        if contribs[j] <= 0 or len(reasons) >= 3:
            break
        text = describe_reason(FEATURE_COLUMNS[j], row)
        if text and text not in reasons:
            reasons.append(text)
    return reasons or ["Combination of weak signals; no single dominant factor"]

@app.route('/score', methods=['POST'])
def score_transaction():
    data = request.json
    df_in = pd.DataFrame([data])
    for f in FEATURE_COLUMNS:
        if f not in df_in.columns:
            df_in[f] = 0.0
    dmatrix = xgb.DMatrix(df_in[FEATURE_COLUMNS])
    contribs = xgb_model.predict(dmatrix, pred_contribs=True)[0]
    xgb_prob = float(xgb_model.predict(dmatrix)[0])
    scaled_nn = scaler.transform(df_in[NN_FEATURES])
    tf_prob = float(tf_model.predict(scaled_nn, verbose=0)[0][0])
    blended = float(best_w[0]*xgb_prob + best_w[1]*tf_prob + best_w[2]*tf_prob)
    tier = "RED" if blended >= red_thr else ("YELLOW" if blended >= yellow_thr else "CLEAR")
    reasons = generate_reasons(contribs, data)
    return jsonify({
        "score": blended,
        "flagged": tier != "CLEAR",
        "action_tier": tier,
        "reasons": reasons
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
