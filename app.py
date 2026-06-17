import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge


load_dotenv()

app = Flask(__name__)
# Allow multipart overhead, then enforce the actual image byte limit ourselves.
app.config["MAX_CONTENT_LENGTH"] = 11 * 1024 * 1024
app.json.ensure_ascii = False

ZHIPU_API_KEY = (
    os.environ.get("ZHIPU_API_KEY")
    or os.environ.get("BIGMODEL_API_KEY")
    or os.environ.get("ZAI_API_KEY")
    or ""
).strip()
ZHIPU_MODEL = os.environ.get("ZHIPU_MODEL", "qwen3-vl-plus").strip()
ZHIPU_ENDPOINT = os.environ.get("ZHIPU_ENDPOINT", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions").strip()

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "").strip()
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
SEARCH_CACHE_TTL_SECONDS = 24 * 60 * 60
MAX_BRAVE_QUERIES_PER_ANALYSIS = 2
SEARCH_CACHE = {}

MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}
FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}

SOURCE_DOMAIN_MAP = {
    "央视": ["cctv.com", "news.cctv.com", "cgtn.com"],
    "央视新闻": ["cctv.com", "news.cctv.com"],
    "中央广播电视总台": ["cctv.com", "news.cctv.com", "cgtn.com"],
    "人民日报": ["people.com.cn", "paper.people.com.cn"],
    "人民网": ["people.com.cn"],
    "新华社": ["news.cn", "xinhuanet.com"],
    "新华网": ["news.cn", "xinhuanet.com"],
    "中国新闻网": ["chinanews.com.cn"],
    "中新社": ["chinanews.com.cn"],
    "澎湃新闻": ["thepaper.cn"],
    "界面新闻": ["jiemian.com"],
    "财新": ["caixin.com"],
    "观察者网": ["guancha.cn"],
    "环球时报": ["huanqiu.com", "globaltimes.cn"],
    "南方周末": ["infzm.com"],
    "第一财经": ["yicai.com"],
    "BBC": ["bbc.com", "bbc.co.uk"],
    "CNN": ["cnn.com"],
    "Reuters": ["reuters.com"],
    "路透": ["reuters.com"],
    "AP": ["apnews.com"],
    "Associated Press": ["apnews.com"],
    "New York Times": ["nytimes.com"],
    "纽约时报": ["nytimes.com"],
    "Washington Post": ["washingtonpost.com"],
    "华盛顿邮报": ["washingtonpost.com"],
    "Bloomberg": ["bloomberg.com"],
    "彭博": ["bloomberg.com"],
    # 财经媒体
    "华尔街见闻": ["wallstreetcn.com"],
    "财联社": ["cls.cn"],
    "证券时报": ["stcn.com"],
    "券商中国": ["stcn.com"],
    "中国基金报": ["chinafundnews.com.cn"],
    "上海证券报": ["cnstock.com"],
    "每日经济新闻": ["nbd.com.cn"],
    "36氪": ["36kr.com"],
    "东方财富": ["eastmoney.com"],
    "同花顺": ["10jqka.com.cn", "hexin.cn"],
}

SOURCE_EXTRACTION_PROMPT = """
你是一名新闻截图信息抽取员。请只从图片中抽取可见信息，不要判断真假。

请返回 JSON，不要返回 Markdown、代码块或 JSON 之外的文字。格式如下：
{
  "has_source": true,
  "source": {
    "name": "媒体/账号/网站/App名称",
    "type": "官方媒体",
    "confidence": "高"
  },
  "title": "图片中最像新闻标题的文字，无法识别则为空字符串",
  "date": "图片中可见发布时间或日期，无法识别则为空字符串",
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "domain_or_app": "图片中可见域名/App/平台名，无法识别则为空字符串"
}

字段约束：
- has_source 为 true/false。
- 如果没有 logo、账号名、域名、App 名、水印等来源标识，has_source=false，其他字段尽量填空。
- source.type 只能是：官方媒体 / 自媒体 / 境外媒体 / 未知
- source.confidence 只能是：高 / 中 / 低
- keywords 给 2 到 6 个中文或英文关键词，优先选择人物、地点、机构、事件名、核心名词。
"""

FINAL_ANALYSIS_PROMPT = """
你是一位专业的新闻真实性核查员。请基于图片内容、抽取信息和外部搜索证据做保守判断。

硬规则：
1. 如果图片没有可识别来源标识，返回 has_source=false。
2. found_official_match=true 时，强烈倾向”可信”；除非图片内容与官网结果明显矛盾，或存在明显篡改/仿冒证据。
3. found_official_match=false 不等于虚假；但如果 queries_used>0 且搜索结果中多个可信来源标题/摘要与截图核心事实一致，可判”可信”；若证据不充分，判”存疑”或”无法判断”。
4. 允许在存在明确反证或明显伪造特征时判”疑似虚假”：包括官方辟谣、来源与内容冲突、仿冒 logo/域名/媒体名、日期异常、截图排版明显伪造、搜索结果显示同一标题只来自低可信搬运/视频/论坛且无权威来源。
5. 如果 queries_used>0、error 为空、但没有官网命中也没有充分可信交叉证据，判”存疑”或”无法判断”。
6. “存疑”必须给出具体可验证疑点；缺少上下文本身不算强疑点。
7. 图片中的新闻发布时间必须和当前美东时间比较；同一天凌晨时间如果早于当前时刻，不是未来时间。
8. 如果外部搜索证据中 error 非空，或 queries_used=0，说明搜索未实际执行。此时禁止仅凭”来源看起来正规”判”可信”，最多判”存疑”或”无法判断”。

请只返回 JSON，不要输出 Markdown、解释文字或代码块。

如果没有来源标识，格式如下：
{
  "has_source": false,
  "message": "图片中未检测到可识别的新闻来源标识，请上传包含媒体 logo、账号名、域名、App 名称或节目水印的截图。"
}

如果有来源标识，格式如下：
{
  "has_source": true,
  "source": {
    "name": "媒体名称",
    "type": "官方媒体",
    "confidence": "高"
  },
  "verdict": "可信",
  "verdict_emoji": "✅",
  "reasons": [
    "具体理由1",
    "具体理由2"
  ],
  "suggestion": "建议用户采取的具体行动",
  "evidence": {
    "official_match_found": true,
    "official_url": "https://example.com/news/xxx",
    "search_queries_used": 1
  }
}

字段约束：
- verdict 只能是：可信 / 存疑 / 疑似虚假 / 无法判断
- verdict_emoji 必须对应：可信=✅，存疑=⚠️，疑似虚假=❌，无法判断=❓
- source.type 只能是：官方媒体 / 自媒体 / 境外媒体 / 未知
- source.confidence 只能是：高 / 中 / 低
- reasons 必须是字符串数组，给出 2 到 5 条简明理由
"""

SOURCE_EXTRACTION_PROMPT_EN = """
You are a news screenshot information extractor. Extract only visible information from the image — do not judge authenticity.

Return only JSON, no Markdown, code blocks, or extra text:
{
  "has_source": true,
  "source": {
    "name": "media/account/website/app name",
    "type": "Official Media",
    "confidence": "High"
  },
  "title": "the most news-headline-like text visible in the image, or empty string",
  "date": "visible publication time/date, or empty string",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "domain_or_app": "visible domain/app/platform name, or empty string"
}

Field constraints:
- has_source: true/false. If no logo, account name, domain, app name, watermark, or other source identifier is visible, set has_source=false; leave other fields empty.
- source.type must be exactly one of: Official Media / Independent Media / Foreign Media / Unknown
- source.confidence must be exactly one of: High / Medium / Low
- keywords: 2 to 6 keywords (Chinese or English); prioritize names of people, places, organizations, and events.
"""

FINAL_ANALYSIS_PROMPT_EN = """
You are a professional news fact-checker. Make a conservative judgment based on the image content, extracted metadata, and external search evidence.

Hard rules:
1. If the image has no identifiable source, return has_source=false.
2. When found_official_match=true, strongly lean toward "Credible" — unless the image content clearly contradicts the official result, or there is clear evidence of tampering or impersonation.
3. found_official_match=false does not mean false news. If queries_used>0 and multiple credible sources in search results align with the screenshot's core facts, you may judge "Credible". Otherwise judge "Suspicious" or "Unable to Determine".
4. You may judge "Likely False" when there is clear counter-evidence or obvious fabrication: official denials, source-content conflict, fake logos/domains/media names, abnormal dates, obvious layout forgery, or search results showing the headline only on low-credibility aggregator sites with no authoritative origin.
5. If queries_used>0, error is empty, but there is neither an official match nor sufficient cross-source credible evidence, judge "Suspicious" or "Unable to Determine".
6. "Suspicious" must cite specific, verifiable doubts. Lack of context alone is not a strong indicator.
7. Compare the publication time visible in the image against the current Eastern Time. An early-morning time earlier than the current moment on the same day is in the past, not the future.
8. If the external search evidence shows error is non-empty or queries_used=0, the search was not actually executed. Do not judge "Credible" solely because the source looks legitimate — judge "Suspicious" or "Unable to Determine" at most.

Return only JSON — no Markdown, explanation, or code blocks.

If no source identifier is present:
{
  "has_source": false,
  "message": "No identifiable news source found in the image. Please upload a screenshot that includes a media logo, account name, domain, app name, or program watermark."
}

If a source identifier is present:
{
  "has_source": true,
  "source": {
    "name": "media name",
    "type": "Official Media",
    "confidence": "High"
  },
  "verdict": "Credible",
  "verdict_emoji": "✅",
  "reasons": [
    "Specific reason 1",
    "Specific reason 2"
  ],
  "suggestion": "Specific action the user should take",
  "evidence": {
    "official_match_found": true,
    "official_url": "https://example.com/news/xxx",
    "search_queries_used": 1
  }
}

Field constraints:
- verdict must be exactly one of: Credible / Suspicious / Likely False / Unable to Determine
- verdict_emoji must match: Credible=✅, Suspicious=⚠️, Likely False=❌, Unable to Determine=❓
- source.type must be exactly one of: Official Media / Independent Media / Foreign Media / Unknown
- source.confidence must be exactly one of: High / Medium / Low
- reasons must be a string array with 2 to 5 concise reasons
"""


def parse_model_json(text: str) -> dict:
    """Parse model JSON, including responses wrapped in code fences."""
    cleaned = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    elif not cleaned.startswith("{"):
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0).strip()

    return json.loads(cleaned)


def normalize_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_cache_key(query: str) -> str:
    return re.sub(r"\s+", " ", query.lower()).strip()


def is_placeholder_key(value: str) -> bool:
    if not value:
        return True
    placeholders = {"你的密钥粘贴在这里", "your-api-key", "your-brave-api-key"}
    return value.strip() in placeholders

def current_time_context() -> str:
    try:
        now = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now = datetime.now(timezone(timedelta(hours=-5)))
    tz_abbr = now.strftime("%Z")  # EST or EDT
    return (
        f"当前日期时间（美东时间 {tz_abbr}）：{now.strftime('%Y-%m-%d %H:%M:%S')}。"
        "所有相对时间、今天、昨天、明天、凌晨、上午、下午、晚上都必须以这个时间为准。"
        "如果图片时间是今天凌晨且早于当前时间，它属于过去，不是未来时间。"
    )


def current_time_context_en() -> str:
    try:
        now = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now = datetime.now(timezone(timedelta(hours=-5)))
    tz_abbr = now.strftime("%Z")
    return (
        f"Current date and time (Eastern Time, {tz_abbr}): {now.strftime('%Y-%m-%d %H:%M:%S')}. "
        "All relative times — today, yesterday, tomorrow, morning, afternoon, evening — must be based on this time. "
        "If the image shows a time early this morning that is earlier than the current moment, it is in the past, not the future."
    )

def validate_image(file_storage):
    if file_storage.mimetype not in ALLOWED_MIME_TYPES:
        return None, "仅支持 JPG、PNG、WEBP 格式的图片。"

    image_bytes = file_storage.read()
    if not image_bytes:
        return None, "上传的图片为空，请重新选择文件。"
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return None, "图片不能超过 10MB，请压缩后再上传。"

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError):
        return None, "图片文件损坏或格式不受支持。"

    if image.format not in ALLOWED_PIL_FORMATS:
        return None, "仅支持 JPG、PNG、WEBP 格式的图片。"

    return {
        "bytes": image_bytes,
        "mime_type": FORMAT_TO_MIME[image.format],
    }, None


def image_part(image_payload: dict) -> dict:
    image_data_url = (
        f"data:{image_payload['mime_type']};base64,"
        f"{base64.b64encode(image_payload['bytes']).decode('ascii')}"
    )
    return {"type": "image_url", "image_url": {"url": image_data_url}}


def call_zhipu_with_prompt(prompt: str, image_payload: dict, extra_text: str = "", time_context: str = "") -> str:
    tc = time_context or current_time_context()
    content = [{"type": "text", "text": tc + "\n\n" + prompt}]
    if extra_text:
        content.append({"type": "text", "text": extra_text})
    content.append(image_part(image_payload))

    payload = {
        "model": ZHIPU_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "stream": False,
    }

    max_retries = 5
    for attempt in range(max_retries):
        # Rebuild request each attempt (Request object is not reusable)
        api_request = urllib.request.Request(
            ZHIPU_ENDPOINT,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {ZHIPU_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(api_request, timeout=60) as response:
                response_body = response.read().decode("utf-8")
            break  # success, exit retry loop
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < max_retries - 1:
                wait = 8 * (attempt + 1)  # 8s, 16s, 24s, 32s
                time.sleep(wait)
                continue
            if exc.code == 429:
                raise RuntimeError("_RATE_LIMIT_") from exc
            raise RuntimeError(f"智谱 API 请求失败（HTTP {exc.code}）：{error_body[:800]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接智谱 API：{exc.reason}") from exc

    data = json.loads(response_body)
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
    return str(content).strip()


def extract_news_metadata(image_payload: dict, lang: str = "zh") -> dict:
    prompt = SOURCE_EXTRACTION_PROMPT_EN if lang == "en" else SOURCE_EXTRACTION_PROMPT
    tc = current_time_context_en() if lang == "en" else current_time_context()
    text = call_zhipu_with_prompt(prompt, image_payload, time_context=tc)
    return parse_model_json(text)


def official_domains_for_source(source_name: str, domain_or_app: str = "") -> list[str]:
    source_name = normalize_text(source_name)
    domain_or_app = normalize_text(domain_or_app).lower()
    domains = []

    for name, mapped_domains in SOURCE_DOMAIN_MAP.items():
        if name.lower() in source_name.lower() or source_name.lower() in name.lower():
            domains.extend(mapped_domains)

    domain_match = re.search(r"([a-z0-9-]+\.)+[a-z]{2,}", domain_or_app)
    if domain_match:
        domains.append(domain_match.group(0))

    seen = set()
    unique_domains = []
    for domain in domains:
        domain = domain.lower().strip()
        if domain and domain not in seen:
            seen.add(domain)
            unique_domains.append(domain)
    return unique_domains


def build_search_queries(metadata: dict) -> tuple[list[str], list[str]]:
    if not metadata.get("has_source"):
        return [], []

    source = metadata.get("source") or {}
    source_name = normalize_text(source.get("name"))
    title = normalize_text(metadata.get("title"))
    keywords = [normalize_text(item) for item in metadata.get("keywords") or []]
    keywords = [item for item in keywords if item]
    domains = official_domains_for_source(source_name, metadata.get("domain_or_app"))

    if len(title) < 6 and len(keywords) < 2:
        return [], domains

    queries = []
    if len(title) >= 6:
        queries.append(title)

    compact_terms = " ".join([source_name, title if len(title) < 28 else "", *keywords[:3]]).strip()
    compact_terms = normalize_text(compact_terms)
    if domains and (title or keywords):
        queries.append(f"site:{domains[0]} {title or ' '.join(keywords[:4])}")
    elif compact_terms:
        queries.append(compact_terms)

    seen = set()
    unique_queries = []
    for query in queries:
        key = normalize_cache_key(query)
        if query and key not in seen:
            seen.add(key)
            unique_queries.append(query)
    return unique_queries[:MAX_BRAVE_QUERIES_PER_ANALYSIS], domains


def brave_search(query: str) -> dict:
    cache_key = normalize_cache_key(query)
    cached = SEARCH_CACHE.get(cache_key)
    now = time.time()
    if cached and now - cached["time"] < SEARCH_CACHE_TTL_SECONDS:
        return {**cached["data"], "from_cache": True}

    params = urllib.parse.urlencode({
        "q": query,
        "count": 5,
        "search_lang": "zh-hans",
        "country": "CN",
        "text_decorations": "false",
        "spellcheck": "true",
    })
    api_request = urllib.request.Request(
        f"{BRAVE_ENDPOINT}?{params}",
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": BRAVE_API_KEY,
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(api_request, timeout=20) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Brave Search 请求失败（HTTP {exc.code}）：{error_body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 Brave Search：{exc.reason}") from exc

    raw = json.loads(response_body)
    results = []
    for item in raw.get("web", {}).get("results", [])[:5]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "description": item.get("description", ""),
        })

    data = {"query": query, "results": results, "from_cache": False}
    SEARCH_CACHE[cache_key] = {"time": now, "data": data}
    return data


def url_matches_domain(url: str, domains: list[str]) -> bool:
    url = str(url or "").lower()
    return any(domain.lower() in url for domain in domains)


def title_overlap_score(title: str, result: dict) -> float:
    title = normalize_text(title).lower()
    if not title:
        return 0.0
    haystack = normalize_text(f"{result.get('title', '')} {result.get('description', '')}").lower()
    tokens = [token for token in re.split(r"[\s，。、《》：:；;,.!?！？\-_/|]+", title) if len(token) >= 2]
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in haystack)
    return hits / max(len(tokens), 1)


def collect_search_evidence(metadata: dict) -> dict:
    evidence = {
        "enabled": bool(BRAVE_API_KEY and not is_placeholder_key(BRAVE_API_KEY)),
        "queries": [],
        "results": [],
        "official_domains": [],
        "found_official_match": False,
        "official_url": "",
        "queries_used": 0,
        "cache_hits": 0,
        "error": "",
    }
    if not evidence["enabled"]:
        evidence["error"] = "未配置 BRAVE_API_KEY，已跳过外部搜索。"
        return evidence

    queries, domains = build_search_queries(metadata)
    evidence["official_domains"] = domains
    if not queries:
        evidence["error"] = "来源、标题或关键词不足，已跳过外部搜索以节省次数。"
        return evidence

    title = normalize_text(metadata.get("title"))
    for query in queries:
        try:
            search_data = brave_search(query)
        except Exception as exc:
            evidence["error"] = str(exc)
            break

        evidence["queries_used"] += 0 if search_data.get("from_cache") else 1
        evidence["cache_hits"] += 1 if search_data.get("from_cache") else 0
        evidence["queries"].append(query)
        evidence["results"].append(search_data)

        if domains:
            for result in search_data.get("results", []):
                if url_matches_domain(result.get("url", ""), domains) and title_overlap_score(title, result) >= 0.35:
                    evidence["found_official_match"] = True
                    evidence["official_url"] = result.get("url", "")
                    return evidence

    return evidence


def final_analysis(image_payload: dict, metadata: dict, search_evidence: dict, lang: str = "zh") -> dict:
    if lang == "en":
        prompt = FINAL_ANALYSIS_PROMPT_EN
        tc = current_time_context_en()
        extra_text = "\n".join([
            "Below is the extracted metadata and external search evidence. Base your judgment strictly on this evidence.",
            "Extracted Metadata:",
            json.dumps(metadata, ensure_ascii=False, indent=2),
            "External Search Evidence:",
            json.dumps(search_evidence, ensure_ascii=False, indent=2),
        ])
    else:
        prompt = FINAL_ANALYSIS_PROMPT
        tc = current_time_context()
        extra_text = "\n".join([
            "以下是第一步从图片抽取的信息和后端外部搜索证据，请严格依据这些证据保守判断。",
            "抽取信息：",
            json.dumps(metadata, ensure_ascii=False, indent=2),
            "外部搜索证据：",
            json.dumps(search_evidence, ensure_ascii=False, indent=2),
        ])
    text = call_zhipu_with_prompt(prompt, image_payload, extra_text, time_context=tc)
    result = parse_model_json(text)

    if isinstance(result, dict) and result.get("has_source"):
        # 强制用真实搜索结果覆盖模型自报的 evidence，防止模型幻觉"找到了官网证据"
        result["evidence"] = {
            "official_match_found": bool(search_evidence.get("found_official_match", False)),
            "official_url": search_evidence.get("official_url", ""),
            "search_queries_used": search_evidence.get("queries_used", 0),
            "search_cache_hits": search_evidence.get("cache_hits", 0),
        }
    return result


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(_error):
    return jsonify({"error": "图片不能超过 10MB，请压缩后再上传。"}), 413


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    lang = request.form.get("lang", "en")
    if lang not in ("zh", "en"):
        lang = "en"

    def err(zh_msg: str, en_msg: str) -> str:
        return en_msg if lang == "en" else zh_msg

    if "image" not in request.files:
        return jsonify({"error": err("没有收到图片文件。", "No image file received.")}), 400

    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({"error": err("没有选择图片文件。", "No image file selected.")}), 400

    image_payload, error = validate_image(file)
    if error:
        # validate_image returns Chinese; map to English if needed
        if lang == "en":
            en_errors = {
                "仅支持 JPG、PNG、WEBP 格式的图片。": "Only JPG, PNG, and WEBP images are supported.",
                "上传的图片为空，请重新选择文件。": "The uploaded image is empty. Please select another file.",
                "图片不能超过 10MB，请压缩后再上传。": "Image must be under 10MB. Please compress and try again.",
                "图片文件损坏或格式不受支持。": "Image file is corrupted or unsupported.",
            }
            error = en_errors.get(error, error)
        return jsonify({"error": error}), 400

    if is_placeholder_key(ZHIPU_API_KEY):
        return jsonify({"error": err(
            "未配置 ZHIPU_API_KEY，请先在 .env 文件中设置智谱 API Key。",
            "ZHIPU_API_KEY is not configured. Please set it in the .env file.",
        )}), 500

    try:
        metadata = extract_news_metadata(image_payload, lang)
        search_evidence = collect_search_evidence(metadata)
        result = final_analysis(image_payload, metadata, search_evidence, lang)
    except json.JSONDecodeError:
        return jsonify({"error": err("AI 返回格式异常，请重试。", "AI returned an unexpected format. Please try again.")}), 502
    except Exception as exc:
        if "_RATE_LIMIT_" in str(exc):
            return jsonify({"error": err(
                "AI 服务暂时繁忙，请等待约 30 秒后再试。",
                "AI service is busy. Please wait about 30 seconds and try again.",
            )}), 429
        return jsonify({"error": err(f"AI 分析失败：{exc}", f"AI analysis failed: {exc}")}), 502

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)