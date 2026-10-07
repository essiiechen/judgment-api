#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
判案 API（FastAPI），监听 127.0.0.1:8000

  GET  /health  -> {"ok": true}                      不需要 key
  POST /judge   -> {"cause", "laws", "direction", "verdict"}
                   需要 Authorization: Bearer <key>，否则 401

内部不调任何大模型：3,520 份判决书建成 BM25 判例库，新案情进来检索最像的判例，
案由/方向用相似度加权投票（方向还乘了案由先验），主文用「同案由 + 同方向」里
最像那份做模板并换上本案当事人。

启动：.venv/bin/python api.py
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from judge import Judge, load          # noqa: E402

API_KEY = os.environ.get("JUDGE_API_KEY", "moot-2026-7f3c9a")
CACHE = ROOT / "judge_index.pkl"

app = FastAPI(title="判案 API", version="1.0")
J = None


def build():
    global J
    t0 = time.time()
    if CACHE.exists():
        try:
            with open(CACHE, "rb") as f:
                J = pickle.load(f)
            print(f"[boot] 索引缓存加载 {time.time()-t0:.1f}s｜{len(J.rec)} 份判例", flush=True)
            return
        except Exception as e:
            print(f"[boot] 缓存不可用（{e}），重建", flush=True)
    recs = load()
    J = Judge(recs)
    try:
        with open(CACHE, "wb") as f:
            pickle.dump(J, f)
    except Exception as e:
        print(f"[boot] 缓存写入失败（{e}），不影响服务", flush=True)
    print(f"[boot] 索引构建 {time.time()-t0:.1f}s｜{len(J.rec)} 份判例", flush=True)


@app.on_event("startup")
def _startup():
    build()


class Req(BaseModel):
    case_text: str = ""


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/judge")
def judge(req: Req, authorization: str | None = Header(default=None)):
    # --- 鉴权 ---
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    if authorization[7:].strip() != API_KEY:
        raise HTTPException(status_code=401, detail="invalid key")

    t0 = time.time()
    try:
        out = J(req.case_text or "")
    except Exception as e:                      # 兜底：宁可给个保守答案，也不要 500
        print(f"[judge] 出错 {e}", flush=True)
        out = {"cause": "", "laws": [], "direction": "部分支持", "verdict": ""}

    res = {
        "cause": out.get("cause") or "其他纠纷",
        "laws": out.get("laws") or ["《中华人民共和国民事诉讼法》第六十四条"],
        "direction": out.get("direction") or "部分支持",
        "verdict": out.get("verdict") or "驳回原告的其他诉讼请求。",
    }
    print(f"[judge] {time.time()-t0:.2f}s cause={res['cause']} dir={res['direction']} "
          f"laws={len(res['laws'])}", flush=True)
    return JSONResponse(res)


@app.exception_handler(HTTPException)
def _h(req, exc):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
