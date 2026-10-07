from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path


class OcrError(RuntimeError):
    pass


PROMPT = """Распознай систему дифференциальных и алгебраических уравнений на изображении.
Верни только уравнения, по одному в строке. Для производных используй dx/dt = ..., для
алгебраических переменных x = .... Сохрани имена переменных и параметров. Не добавляй пояснений."""


def recognize_image(path: str | Path, backend: str = "auto", api_key: str = "") -> str:
    path = Path(path)
    if not path.is_file():
        raise OcrError("Файл изображения не найден")
    if backend in {"auto", "pix2tex"}:
        try:
            from PIL import Image
            from pix2tex.cli import LatexOCR
            return str(LatexOCR()(Image.open(path)))
        except Exception as exc:
            if backend == "pix2tex":
                raise OcrError(f"pix2tex недоступен: {exc}") from exc
    if backend in {"auto", "openai"}:
        key = api_key.strip() or os.getenv("OPENAI_API_KEY", "")
        if key:
            return _recognize_openai(path, key)
        if backend == "openai":
            raise OcrError("Укажите OPENAI_API_KEY или ключ в настройках")
    raise OcrError("Нет OCR: установите pix2tex или задайте OpenAI API key")


def _recognize_openai(path: Path, api_key: str) -> str:
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(path.suffix.lower(), "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    body = {
        "model": "gpt-4.1-mini",
        "store": False,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": PROMPT},
            {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}", "detail": "high"},
        ]}],
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise OcrError(f"Ошибка OpenAI OCR: {exc}") from exc
    if data.get("output_text"):
        return data["output_text"]
    chunks = [
        content.get("text", "")
        for item in data.get("output", [])
        for content in item.get("content", [])
        if content.get("type") == "output_text"
    ]
    if not chunks:
        raise OcrError("OCR не вернул текст")
    return "\n".join(chunks)
