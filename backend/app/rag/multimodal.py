# -*- coding: utf-8 -*-
"""多模态解析：从 PDF/DOCX 提取内嵌图片，调用 DashScope 视觉模型做图表 OCR，
识别文本返回给流水线切块入库；任何单张图片失败只记录错误，不阻断整体摄入。"""
import base64
import io
import os
import zipfile
from typing import Any, Dict, List

import pymupdf
import pytesseract
from openai import OpenAI
from PIL import Image

from app.core.config import settings

OCR_PROMPT = (
    "你是金融文档图表识别助手。请识别这张图片中的全部文字、数字、单位与表格内容，"
    "按从左到右、从上到下的阅读顺序输出；表格用 Markdown 表格表示；"
    "图表中的坐标轴、图例、数据标签数值必须保留；不要解释图片，不要输出识别以外的内容。"
)


def _client() -> OpenAI:
    return OpenAI(api_key=settings.DASHSCOPE_API_KEY, base_url=settings.DASHSCOPE_BASE_URL)


def _is_large_enough(data: bytes) -> bool:
    """过滤小图标/分割线：最短边低于阈值的图片跳过。"""
    try:
        im = Image.open(io.BytesIO(data))
        return min(im.size) >= settings.OCR_MIN_SIDE
    except Exception:
        return False


def extract_images_from_pdf(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    doc = pymupdf.open(path)
    for pno, page in enumerate(doc):
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen:
                continue
            seen.add(xref)
            d = doc.extract_image(xref)
            data = d["image"]
            ext = d["ext"]
            if len(data) < settings.OCR_MIN_IMAGE_BYTES or not _is_large_enough(data):
                continue
            mime = "image/jpeg" if ext.lower() in ("jpg", "jpeg") else f"image/{ext.lower()}"
            out.append({"name": f"page{pno + 1}_img{xref}.{ext}", "mime": mime, "bytes": data})
    return out


FIGURE_CAPTIONS = ("来源：", "数据来源：", "资料来源：")
FIGURE_HEIGHT_PT = 210
FIGURE_X0, FIGURE_X1 = 40, 555


def extract_figure_regions_pdf(path: str) -> List[Dict[str, Any]]:
    """矢量图表没有内嵌位图：按图表注释定位，把注释上方区域渲染成位图再 OCR。"""
    out: List[Dict[str, Any]] = []
    doc = pymupdf.open(path)
    seen_rects = []
    for pno, page in enumerate(doc):
        for cap in FIGURE_CAPTIONS:
            hits = page.search_for(cap)
            if hits:
                break
        if not hits:
            continue
        r = hits[0]
        clip = pymupdf.Rect(FIGURE_X0, max(0, r.y0 - FIGURE_HEIGHT_PT),
                            FIGURE_X1, r.y0 - 3)
        # 与已收集区域大面积重叠则跳过（同页多注释指向同一图）
        if any(abs(clip.y0 - y0) < 30 and pno == p2 for p2, y0 in seen_rects):
            continue
        seen_rects.append((pno, clip.y0))
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2.5, 2.5), clip=clip)
        out.append({"name": f"page{pno + 1}_figure.png", "mime": "image/png",
                    "bytes": pix.tobytes("png")})
    return out


def extract_images_from_docx(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.startswith("word/media/"):
                continue
            data = z.read(name)
            if len(data) < settings.OCR_MIN_IMAGE_BYTES or not _is_large_enough(data):
                continue
            ext = name.split(".")[-1].lower()
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            out.append({"name": os.path.basename(name), "mime": mime, "bytes": data})
    return out


def ocr_image(item: Dict[str, Any]) -> str:
    b64 = base64.b64encode(item["bytes"]).decode()
    resp = _client().chat.completions.create(
        model=settings.OCR_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{item['mime']};base64,{b64}"}},
                ],
            }
        ],
        temperature=0.01,
        timeout=120,
    )
    return resp.choices[0].message.content or ""


def ocr_image_local(item: Dict[str, Any]) -> str:
    """本地 OCR 降级：Tesseract（chi_sim+eng），无需云端额度。"""
    image = Image.open(io.BytesIO(item["bytes"]))
    return pytesseract.image_to_string(image, lang="chi_sim+eng")


def process_file(path: str) -> List[Dict[str, Any]]:
    """提取并逐张 OCR，返回 [{name, ocr_text, error?}]。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        images = extract_images_from_pdf(path) + extract_figure_regions_pdf(path)
    elif ext == ".docx":
        images = extract_images_from_docx(path)
    else:
        return []

    results = []
    for im in images:
        rec = {"name": im["name"], "ocr_text": "", "engine": ""}
        try:
            rec["ocr_text"] = ocr_image(im)
            rec["engine"] = settings.OCR_MODEL
        except Exception as e:
            rec["cloud_error"] = str(e)[:200]
            if settings.OCR_LOCAL_FALLBACK:
                try:
                    rec["ocr_text"] = ocr_image_local(im)
                    rec["engine"] = "tesseract-local"
                except Exception as e2:
                    rec["local_error"] = str(e2)[:200]
        results.append(rec)
    return results
