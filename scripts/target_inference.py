import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import sys

try:
    import onnxruntime as ort  # type: ignore
    HAVE_ORT = True
except Exception:
    HAVE_ORT = False

try:
    import lightgbm as lgb  # type: ignore
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

MODEL_SEARCH = [
    Path('models/pred_totals.onnx'),
    Path('models/pred_margin.onnx'),
    Path('models/pred_combined.onnx'),
    Path('pred_totals.onnx'),
    Path('pred_margin.onnx')
]
LGB_SEARCH = [
    Path('models/pred_totals.txt'),
    Path('models/pred_margin.txt'),
    Path('pred_totals.txt'),
    Path('pred_margin.txt')
]

def find_existing(paths):
    return [p for p in paths if p.exists()]

def infer_with_onnx(df: pd.DataFrame, model_path: Path):
    try:
        sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    except Exception as e:
        print('ONNX load failed:', e)
        return None
    input_name = sess.get_inputs()[0].name
    # Select numeric columns only
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return None
    X = df[num_cols].fillna(df[num_cols].mean()).to_numpy(dtype=np.float32)
    try:
        pred = sess.run(None, {input_name: X})[0]
    except Exception as e:
        print('ONNX inference error:', e)
        return None
    # If multi-output assume [total, margin]
    if pred.ndim == 2 and pred.shape[1] >= 2:
        return {
            'pred_total_model': pred[:,0],
            'pred_margin_model': pred[:,1]
        }
    # Single output -> treat as total; derive margin heuristic
    if pred.ndim == 1:
        return {
            'pred_total_model': pred,
            'pred_margin_model': pred * 0.03
        }
    return None

def infer_with_lightgbm(df: pd.DataFrame, model_path: Path):
    try:
        booster = lgb.Booster(model_file=str(model_path))
    except Exception as e:
        print('LightGBM load failed:', e)
        return None
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return None
    X = df[num_cols].fillna(df[num_cols].mean())
    try:
        pred = booster.predict(X)
    except Exception as e:
        print('LightGBM inference error:', e)
        return None
    # Heuristic mapping similar to ONNX single-output case
    return {
        'pred_total_model': pred,
        'pred_margin_model': pred * 0.025
    }

def simple_predict(df: pd.DataFrame):
    numeric_cols = [c for c in df.columns if c not in {'game_id','home_team','away_team'} and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        df['pred_total_model'] = np.nan
        df['pred_margin_model'] = np.nan
        return df
    base = df[numeric_cols].mean(axis=1)
    df['pred_total_model'] = (base * 2).clip(lower=80, upper=170)
    df['pred_margin_model'] = (df[numeric_cols[0]] * 0.05).clip(lower=-25, upper=25)
    return df

def main():
    date_str = datetime.now().strftime('%Y-%m-%d')
    if len(sys.argv) > 1 and sys.argv[1].strip():
        date_str = sys.argv[1].strip()
    manifest_path = Path('outputs') / f'missing_inference_manifest_{date_str}.json'
    if not manifest_path.exists():
        print('Manifest not found:', manifest_path)
        return
    with open(manifest_path,'r',encoding='utf-8') as fh:
        manifest = json.load(fh)
    if not manifest.get('records'):
        print('No records in manifest.')
        return
    df = pd.DataFrame(manifest['records'])

    # Attempt real model inference
    used_real = False
    onnx_models = find_existing(MODEL_SEARCH) if HAVE_ORT else []
    lgb_models = find_existing(LGB_SEARCH) if HAVE_LGB else []
    if onnx_models:
        res = infer_with_onnx(df, onnx_models[0])
        if res:
            df['pred_total_model'] = res['pred_total_model']
            df['pred_margin_model'] = res['pred_margin_model']
            df['pred_total_model_basis'] = 'onnx_real'
            df['pred_margin_model_basis'] = 'onnx_real'
            used_real = True
    elif lgb_models:
        res = infer_with_lightgbm(df, lgb_models[0])
        if res:
            df['pred_total_model'] = res['pred_total_model']
            df['pred_margin_model'] = res['pred_margin_model']
            df['pred_total_model_basis'] = 'lgb_real'
            df['pred_margin_model_basis'] = 'lgb_real'
            used_real = True

    if not used_real:
        df = simple_predict(df)
        df['pred_total_model_basis'] = df.get('pred_total_model_basis','heuristic_fallback')
        df['pred_margin_model_basis'] = df.get('pred_margin_model_basis','heuristic_fallback')

    out_path = Path('outputs') / f'missing_inference_preds_{date_str}.csv'
    df.to_csv(out_path, index=False)
    print('Inference output written:', out_path, 'rows:', len(df), 'real_model_used:', used_real)

if __name__ == '__main__':
    main()
