"""
app/main.py — Expresso Churn Prediction API
=============================================
FastAPI backend với 3 endpoints:
    POST /predict        — predict 1 khách hàng (JSON)
    POST /predict-batch  — predict hàng loạt (CSV upload)
    POST /explain        — Gemini 2.0 Flash sinh nhận định tự nhiên

Cách chạy:
    cd expresso-churn
    export GEMINI_API_KEY=AIza...
    uvicorn app.main:app --reload --port 8000

Swagger UI: http://localhost:8000/docs
"""

import io
import json
import os

from google import genai
from google.genai import types
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional

from app.model import (
    VALID_REGIONS,
    VALID_TENURES,
    predict_batch,
    predict_single,
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title       = "Expresso Churn Prediction API",
    description = "Dự đoán khả năng khách hàng rời mạng Expresso (Senegal)",
    version     = "1.0.0",
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class CustomerInput(BaseModel):
    REGION         : Optional[str]   = Field(None, example="DAKAR")
    TENURE         : Optional[str]   = Field(None, example="K > 24 month")
    MONTANT        : Optional[float] = Field(None, example=4200.0)
    FREQUENCE_RECH : Optional[float] = Field(None, example=8.0)
    REVENUE        : Optional[float] = Field(None, example=4199.0)
    FREQUENCE      : Optional[float] = Field(None, example=14.0)
    DATA_VOLUME    : Optional[float] = Field(None, example=1.0)
    ON_NET         : Optional[float] = Field(None, example=314.0)
    ORANGE         : Optional[float] = Field(None, example=132.0)
    TIGO           : Optional[float] = Field(None, example=None)
    REGULARITY     : Optional[int]   = Field(None, example=20)
    TOP_PACK       : Optional[str]   = Field(None, example="On net 200F=Unlimited _call24H")
    FREQ_TOP_PACK  : Optional[float] = Field(None, example=3.0)

    class Config:
        extra = "allow"


class ExplainInput(BaseModel):
    """Input cho /explain — dữ liệu khách hàng + kết quả predict."""
    customer   : dict  # raw input từ form
    prediction : dict  # output từ /predict


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/health")
def health():
    return {
        "status"   : "ok",
        "model"    : "CatBoost Expresso Churn v1.0",
        "gemini"   : "configured" if GEMINI_API_KEY else "not configured (fallback mode)",
    }


@app.get("/schema")
def schema():
    return {"regions": VALID_REGIONS, "tenures": VALID_TENURES}


@app.post("/predict")
def predict(customer: CustomerInput):
    try:
        result = predict_single(customer.model_dump())
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict-batch")
async def predict_batch_endpoint(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file .csv")
    try:
        csv_bytes          = await file.read()
        result_df, summary = predict_batch(csv_bytes)
        output = io.StringIO()
        result_df.to_csv(output, index=False)
        output.seek(0)
        filename = file.filename.replace(".csv", "_predictions.csv")
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode("utf-8")),
            media_type = "text/csv",
            headers    = {
                "Content-Disposition"           : f'attachment; filename="{filename}"',
                "X-Summary"                     : str(summary),
                "Access-Control-Expose-Headers" : "X-Summary",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/explain")
async def explain(body: ExplainInput):
    """
    Dùng Gemini 2.0 Flash sinh nhận định tự nhiên từ kết quả predict.
    Nếu không có GEMINI_API_KEY → trả 503 để frontend tự fallback hardcode.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY chưa được cấu hình.",
        )

    prob       = body.prediction.get("churn_probability", 0)
    risk_level = body.prediction.get("risk_level", "")
    inp        = body.customer

    # Map field kỹ thuật → (tên dễ hiểu, giá trị, ngưỡng tham chiếu từ EDA)
    fields_vi = {
        "REGION"        : ("Khu vực sinh sống",
                           inp.get("REGION"),
                           "Vùng địa lý tại Senegal. Một số vùng có tỉ lệ rời mạng cao hơn (VD: SEDHIOU cao nhất, KAFFRINE thấp nhất)"),
        "TENURE"        : ("Thâm niên dùng mạng",
                           inp.get("TENURE"),
                           "Thang: D(3-6th) → K(>24th). Dưới 12 tháng rủi ro cao hơn. Trên 24 tháng thường trung thành"),
        "REGULARITY"    : ("Số ngày có hoạt động trong 90 ngày gần nhất",
                           inp.get("REGULARITY"),
                           "Thang 1–62 ngày. Khách KHÔNG rời mạng: trung bình 34 ngày. Khách SẮP rời: trung bình chỉ 2 ngày. Dưới 10 là rất đáng lo"),
        "MONTANT"       : ("Tổng tiền đã nạp vào tài khoản",
                           inp.get("MONTANT"),
                           "Đơn vị CFA Franc (1.000 CFA ≈ 40.000 VND). Khách bình thường: ~3.075 CFA/kỳ. Khách sắp rời: ~1.000 CFA"),
        "FREQUENCE_RECH": ("Số lần nạp tiền trong kỳ",
                           inp.get("FREQUENCE_RECH"),
                           "Khách bình thường nạp ~7 lần/kỳ. Khách sắp rời chỉ ~2 lần. Dưới 3 lần là dấu hiệu xấu"),
        "REVENUE"       : ("Doanh thu nhà mạng thu được từ khách",
                           inp.get("REVENUE"),
                           "Đơn vị CFA. Khách bình thường: ~3.000 CFA. Khách sắp rời: ~800 CFA. Dưới 1.000 là đáng lo"),
        "FREQUENCE"     : ("Tổng số lần phát sinh giao dịch (gọi/nhắn tin/data)",
                           inp.get("FREQUENCE"),
                           "Khách bình thường: ~9 lần/kỳ. Khách sắp rời: ~3 lần. Dưới 4 lần = gần như không dùng dịch vụ"),
        "DATA_VOLUME"   : ("Lượng data internet đã dùng",
                           inp.get("DATA_VOLUME"),
                           "Đơn vị MB. Khách bình thường: ~261 MB. Khách sắp rời: ~82 MB. Không có dữ liệu = khách không dùng data"),
        "ON_NET"        : ("Số cuộc gọi trong mạng Expresso",
                           inp.get("ON_NET"),
                           "Khách bình thường: ~29 cuộc. Khách sắp rời: ~5 cuộc. Rất thấp có thể đang dùng mạng khác thay thế"),
        "ORANGE"        : ("Số cuộc gọi sang mạng Orange (đối thủ lớn nhất tại Senegal)",
                           inp.get("ORANGE"),
                           "Orange là mạng cạnh tranh chính. Gọi sang Orange nhiều là bình thường, không có nghĩa là sắp rời"),
        "TIGO"          : ("Số cuộc gọi sang mạng Tigo",
                           inp.get("TIGO"),
                           "Tigo là mạng nhỏ hơn. ~60% khách không có dữ liệu Tigo — hoàn toàn bình thường"),
    }

    # Chỉ đưa vào prompt các field có dữ liệu thực
    customer_text = "\n".join(
        f"- {label}: {val} | Tham chiếu: {context}"
        for _, (label, val, context) in fields_vi.items()
        if val is not None
    )
    if not customer_text:
        customer_text = "Không có dữ liệu cụ thể nào được cung cấp."

    prompt = f"""Bạn là chuyên gia phân tích khách hàng viễn thông tại Senegal, làm việc cho nhà mạng Expresso.

THÔNG TIN KHÁCH HÀNG (kèm ngưỡng so sánh):
{customer_text}

KẾT QUẢ MÔ HÌNH AI (CatBoost, AUC 0.92):
- Xác suất rời mạng: {round(prob * 100, 1)}%
- Mức độ rủi ro: {risk_level}

Lưu ý quan trọng: Mô hình bị ảnh hưởng nhiều bởi yếu tố địa lý (43%) và thâm niên (ảnh hưởng lớn).
Vì vậy % có thể thấp dù hành vi xấu (do vùng địa lý lịch sử ít rời mạng), hoặc ngược lại.

Hãy phân tích TÁCH BIỆT 2 góc nhìn:

1. TẠI SAO MÔ HÌNH DỰ ĐOÁN NHƯ VẬY: Giải thích ngắn gọn dựa trên yếu tố địa lý và thâm niên
2. THỰC TRẠNG HÀNH VI: Phân tích từng chỉ số hành vi (tiền nạp, giao dịch, hoạt động...) SO VỚI NGƯỠNG — trung thực, không bị ảnh hưởng bởi %
3. KẾT LUẬN: Tổng hợp 2 góc nhìn, gợi ý hành động phù hợp

Quy tắc bắt buộc:
- KHÔNG dùng tên biến kỹ thuật (REGULARITY, MONTANT...)
- Chỉ nhận xét chỉ số có dữ liệu thực
- behavior_signals phải trung thực với data — nếu xấu thì nói xấu
- type của mỗi signal: "ok" (tốt hơn ngưỡng), "warn" (trung bình), "bad" (đáng lo)

Trả về JSON, không thêm bất kỳ text nào ngoài JSON:
{{
  "model_reason": "1-2 câu giải thích tại sao mô hình cho kết quả % này (địa lý, thâm niên...)",
  "behavior_signals": [
    {{"type": "ok|warn|bad", "text": "nhận xét chỉ số kèm con số cụ thể và so sánh ngưỡng"}},
    {{"type": "ok|warn|bad", "text": "..."}}
  ],
  "conclusion": "1-2 câu kết luận tổng hợp và gợi ý hành động cụ thể"
}}"""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}"
    )

    try:
        client  = genai.Client(api_key=GEMINI_API_KEY)
        resp    = await client.aio.models.generate_content(
            model  = "gemini-2.5-flash-lite",
            contents = prompt,
            config = types.GenerateContentConfig(
                temperature        = 0.2,
                max_output_tokens  = 1200,
                response_mime_type = "application/json",
            ),
        )

        text = resp.text.strip()
        print(f"[/explain] Gemini OK — {text[:80]}...")

        result = json.loads(text)

        if not all(k in result for k in ("model_reason", "behavior_signals", "conclusion")):
            raise ValueError(f"JSON thiếu fields, có: {list(result.keys())}")

        return JSONResponse(content=result)

    except json.JSONDecodeError as e:
        print(f"[/explain] JSON decode error: {e} — text: {text[:200]}")
        raise HTTPException(status_code=502, detail="Gemini trả về JSON không hợp lệ")
    except Exception as e:
        print(f"[/explain] Error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Explain error: {str(e)}")