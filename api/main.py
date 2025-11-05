from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os

from inference.engine import load_inference_engine


class PredictRequest(BaseModel):
    data: List[List[float]]  # shape: (timesteps, channels=8)
    top_k: Optional[int] = 5
    normalize: Optional[bool] = True


app = FastAPI(title="SignGlove Inference API", version="1.0.0")


# Lazy singleton for engine
_engine = None
_latest = {"ts": None, "result": None}


def get_engine():
    global _engine
    if _engine is None:
        model_path = os.environ.get(
            "SIGNGLOVE_MODEL_PATH",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "best_model", "ms3dgru_best.ckpt")),
        )
        model_type = os.environ.get("SIGNGLOVE_MODEL_TYPE", "MS3DGRU")
        device = os.environ.get("SIGNGLOVE_DEVICE", "cpu")
        scaler_path = os.environ.get("SIGNGLOVE_SCALER_PATH", None)
        # If scaler is not provided next to the model, engine will warn and continue
        _engine = load_inference_engine(
            model_path=model_path,
            model_type=model_type,
            device=device,
            scaler_path=scaler_path,
            single_predict_device=device,
        )
    return _engine


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <title>SignGlove Inference Demo</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
      textarea { width: 100%; height: 220px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
      pre { background: #f6f8fa; padding: 12px; border-radius: 6px; overflow: auto; }
      .row { display: flex; gap: 12px; align-items: center; }
      .badge { display:inline-block; padding: 10px 14px; border-radius: 10px; background:#111; color:#fff; font-size: 28px; font-weight: 700; }
      table { border-collapse: collapse; }
      td, th { border:1px solid #ddd; padding: 6px 10px; }
    </style>
  </head>
  <body>
    <h2>SignGlove Inference Demo</h2>
    <p>JSON으로 (timesteps x 8) 센서 배열을 입력하고 Predict를 눌러보세요.</p>
    <div class=\"row\">
      <button id=\"btn-sample\">샘플 넣기</button>
      <label>top_k <input id=\"topk\" type=\"number\" value=\"5\" min=\"1\" max=\"24\"/></label>
      <label><input id=\"normalize\" type=\"checkbox\" checked/> normalize</label>
      <button id=\"btn-run\">Predict</button>
    </div>
    <p></p>
    <textarea id=\"input\" placeholder=\"[[t0_ch0,...,t0_ch7], [t1_ch0,...,t1_ch7], ...]\"></textarea>
    <h3>Top-1</h3>
    <div id=\"top1\" class=\"badge\">-</div>
    <h3>Top-K</h3>
    <table id=\"topkTable\"><thead><tr><th>Class</th><th>Prob</th></tr></thead><tbody></tbody></table>
    <h3>Raw</h3>
    <pre id=\"out\"></pre>
    <script>
      const ta = document.getElementById('input');
      const out = document.getElementById('out');
      const top1 = document.getElementById('top1');
      const tbody = document.querySelector('#topkTable tbody');
      document.getElementById('btn-sample').onclick = () => {
        // 87x8 zero sample for quick test
        const steps = 87, ch = 8; const arr = Array.from({length: steps}, () => Array(ch).fill(0));
        ta.value = JSON.stringify(arr, null, 2);
      };
      document.getElementById('btn-run').onclick = async () => {
        try {
          const topk = parseInt(document.getElementById('topk').value || '5', 10);
          const normalize = document.getElementById('normalize').checked;
          const data = JSON.parse(ta.value);
          const res = await fetch('/predict', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ data, top_k: topk, normalize })
          });
          const json = await res.json();
          out.textContent = JSON.stringify(json, null, 2);
          // update pretty views
          const pred = json.predicted_class || (json.top_k_predictions && json.top_k_predictions[0] && json.top_k_predictions[0].class_name) || '-';
          top1.textContent = pred;
          tbody.innerHTML = '';
          (json.top_k_predictions || []).forEach(item => {
            const tr = document.createElement('tr');
            const tdC = document.createElement('td');
            const tdP = document.createElement('td');
            tdC.textContent = item.class_name ?? item.class_idx ?? '?';
            tdP.textContent = (item.probability != null ? (item.probability*100).toFixed(1)+'%' : '-');
            tr.appendChild(tdC); tr.appendChild(tdP);
            tbody.appendChild(tr);
          });
        } catch (e) {
          out.textContent = 'Error: ' + e;
        }
      };
    </script>
  </body>
</html>
"""


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        engine = get_engine()
        result = engine.predict_single(req.data, top_k=req.top_k or 5, return_all_info=True, normalize=bool(req.normalize))
        # lightweight server-side log
        try:
            import json, datetime, pathlib
            logs_dir = pathlib.Path(__file__).resolve().parent.parent / 'logs'
            logs_dir.mkdir(parents=True, exist_ok=True)
            with open(logs_dir / 'predictions.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'ts': datetime.datetime.utcnow().isoformat()+'Z',
                    'top1': result.get('predicted_class'),
                    'topk': result.get('top_k_predictions', [])
                }, ensure_ascii=False) + '\n')
            # store latest in-memory
            _latest['ts'] = datetime.datetime.utcnow().isoformat()+'Z'
            _latest['result'] = result
        except Exception:
            pass
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    # Touch engine to validate model presence
    _ = get_engine()
    return {"status": "ok"}


@app.get("/latest")
def latest():
    if not _latest["result"]:
        return {"ts": None, "result": None}
    return _latest


@app.get("/live", response_class=HTMLResponse)
def live_page():
    return """
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <title>SignGlove Live</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
      .big { font-size: 64px; font-weight: 800; }
      .muted { color: #666; }
      table { border-collapse: collapse; margin-top: 16px; }
      td, th { border:1px solid #ddd; padding:8px 12px; }
    </style>
  </head>
  <body>
    <h2>SignGlove Live</h2>
    <div class=\"muted\">자동 갱신 (0.2s)</div>
    <div id=\"top1\" class=\"big\">-</div>
    <div class=\"muted\">updated: <span id=\"ts\">-</span></div>
    <table>
      <thead><tr><th>Class</th><th>Prob</th></tr></thead>
      <tbody id=\"tbody\"></tbody>
    </table>
    <script>
      const top1 = document.getElementById('top1');
      const ts = document.getElementById('ts');
      const tbody = document.getElementById('tbody');
      async function tick(){
        try{
          const r = await fetch('/latest');
          const j = await r.json();
          ts.textContent = j.ts || '-';
          const res = j.result || {};
          const pred = res.predicted_class || (res.top_k_predictions && res.top_k_predictions[0] && res.top_k_predictions[0].class_name) || '-';
          top1.textContent = pred;
          tbody.innerHTML = '';
          (res.top_k_predictions || []).forEach(it => {
            const tr = document.createElement('tr');
            const tdC = document.createElement('td');
            const tdP = document.createElement('td');
            tdC.textContent = it.class_name ?? it.class_idx ?? '?';
            tdP.textContent = (it.probability!=null?(it.probability*100).toFixed(1)+'%':'-');
            tr.appendChild(tdC); tr.appendChild(tdP); tbody.appendChild(tr);
          });
        }catch(e){ /* noop */ }
      }
      setInterval(tick, 200);
      tick();
    </script>
  </body>
</html>
"""


