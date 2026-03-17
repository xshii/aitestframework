"""REST API routes for golden data management.

Model → Version → Operator → Data Items (category/shape/layout/precision).
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time

from flask import Blueprint, current_app, jsonify, request, send_file, send_from_directory

logger = logging.getLogger(__name__)

from aitf.ds.store import (
    ALLOWED_EXT,
    CATEGORY_CHOICES,
    LAYOUT_CHOICES,
    PRECISION_CHOICES,
    DataItem,
    GoldenStore,
    OperatorMeta,
    build_golden_filename,
    parse_golden_filename,
)
from aitf.web.extensions import get_golden_store as _store

bp = Blueprint("datastore", __name__, template_folder="templates")


def _client_ip() -> str:
    return request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")


# -- listing ----------------------------------------------------------------

@bp.route("/api/golden", methods=["GET"])
def list_golden():
    model_filter = request.args.get("model")
    entries = _store().list(model=model_filter)
    return jsonify([
        {
            "model": e.model,
            "version": e.version,
            "operator": e.operator,
            "meta": e.meta.to_dict(),
            "files": e.files,
            "file_sizes": e.file_sizes,
            "total_size": e.total_size,
        }
        for e in entries
    ])


@bp.route("/api/golden/models", methods=["GET"])
def list_models():
    return jsonify(_store().list_models())


@bp.route("/api/golden/<model>/versions", methods=["GET"])
def list_versions(model):
    return jsonify(_store().list_versions(model))


@bp.route("/api/golden/<model>/<version>/operators", methods=["GET"])
def list_operators(model, version):
    return jsonify(_store().list_operators(model, version))


@bp.route("/api/golden/<model>/<version>/info", methods=["GET"])
def version_info(model, version):
    info = _store().version_info(model, version)
    if not info:
        return jsonify({"error": "not found"}), 404
    return jsonify(info)


@bp.route("/api/golden/<model>/<version>/reorder", methods=["PUT"])
def reorder_operators(model, version):
    """Update operator display order."""
    data = request.get_json(silent=True) or {}
    order = data.get("order", [])
    if not isinstance(order, list):
        return jsonify({"error": "order must be an array"}), 400
    store = _store()
    store.reorder_operators(model, version, order)
    store.log("reorder_operators", model=model, version=version, ip=_client_ip())
    return jsonify({"order": order})


_overview_stats_cache: dict = {"ts": 0, "data": {}}
_overview_stats_lock = threading.Lock()


@bp.route("/api/golden/overview-stats", methods=["GET"])
def get_overview_stats():
    """Return cached overview stats (missing file counts per version)."""
    return jsonify(_overview_stats_cache)


@bp.route("/api/golden/overview-stats", methods=["POST"])
def refresh_overview_stats():
    """Recompute overview stats: missing file count per model/version.

    Rate-limited to at most once per 60 seconds globally.
    """
    # Non-blocking: if another thread is computing, return cached
    if not _overview_stats_lock.acquire(blocking=False):
        return jsonify({"error": "统计正在进行中，请稍后查看",
                        "ts": _overview_stats_cache["ts"],
                        "data": _overview_stats_cache["data"]}), 429
    try:
        elapsed = time.time() - _overview_stats_cache["ts"]
        if elapsed < 60:
            remaining = int(60 - elapsed)
            return jsonify({"error": f"统计冷却中，请 {remaining} 秒后再试",
                            "ts": _overview_stats_cache["ts"],
                            "data": _overview_stats_cache["data"]}), 429
        entries = _store().list()
        ver_stats: dict[str, dict[str, int]] = {}
        for e in entries:
            key = f"{e.model}||{e.version}"
            if key not in ver_stats:
                ver_stats[key] = {"missing": 0}
            file_set = set(e.files)
            for d in e.meta.data:
                has_match = False
                for f in file_set:
                    parsed = parse_golden_filename(f)
                    if parsed and parsed["category"] == d.category and parsed["seq"] == d.seq:
                        has_match = True
                        break
                if not has_match:
                    ver_stats[key]["missing"] += 1
        _overview_stats_cache["ts"] = time.time()
        _overview_stats_cache["data"] = ver_stats
        return jsonify(_overview_stats_cache)
    finally:
        _overview_stats_lock.release()


@bp.route("/api/golden/search", methods=["GET"])
def search_operators():
    """Search operators by name across all models."""
    q = (request.args.get("q") or "").strip().lower()
    if not q:
        return jsonify([])
    entries = _store().list()
    return jsonify([
        {
            "model": e.model,
            "version": e.version,
            "operator": e.operator,
        }
        for e in entries
        if q in e.operator.lower()
    ])


@bp.route("/api/golden/choices", methods=["GET"])
def get_choices():
    return jsonify({
        "category": list(CATEGORY_CHOICES),
        "precision": list(PRECISION_CHOICES),
        "layout": list(LAYOUT_CHOICES),
        "filename_format": "{op}_{category}{seq}_{layout}_{precision}_{shape}.txt",
        "filename_example": "matmul_input0_NCHW_FP16_128x96.txt",
        "allowed_ext": list(ALLOWED_EXT),
    })


@bp.route("/api/golden/parse-filename", methods=["POST"])
def parse_filename():
    """Parse a golden filename and return extracted metadata."""
    data = request.get_json(silent=True) or {}
    filename = data.get("filename", "")
    result = parse_golden_filename(filename)
    if not result:
        return jsonify({"error": "无法解析文件名，格式应为: {op}_{category}{seq}_{layout}_{precision}_{shape}.txt"}), 400
    return jsonify(result)


# -- metadata ---------------------------------------------------------------

@bp.route("/api/golden/<model>/<version>/<operator>/meta", methods=["GET"])
def get_meta(model, version, operator):
    store = _store()
    ops = store.list_operators(model, version)
    if operator not in ops:
        return jsonify({"error": "not found"}), 404
    meta = store.load_meta(model, version, operator)
    return jsonify(meta.to_dict())


@bp.route("/api/golden/<model>/<version>/<operator>/meta", methods=["PUT"])
def update_meta(model, version, operator):
    """Replace operator metadata (name + data items list)."""
    raw = request.get_json(silent=True) or {}
    raw.setdefault("name", operator)
    meta = OperatorMeta.from_dict(raw)
    store = _store()
    store.save_meta(model, version, operator, meta)
    store.log("update_meta", model=model, version=version, operator=operator, ip=_client_ip())
    return jsonify(meta.to_dict())


@bp.route("/api/golden/<model>/<version>/<operator>/data-item", methods=["POST"])
def add_data_item(model, version, operator):
    """Append a data item to the operator. Uses provided seq, or auto-assigns."""
    raw = request.get_json(silent=True) or {}
    item = DataItem.from_dict(raw)
    store = _store()
    meta = store.load_meta(model, version, operator)
    if "seq" not in raw:
        next_seq = max((d.seq for d in meta.data), default=-1) + 1
        item.seq = next_seq
    # 同一算子同一种类下序号不能重复
    for d in meta.data:
        if d.category == item.category and d.seq == item.seq:
            return jsonify({"error": f"序号重复: {item.category}{item.seq} 已存在"}), 409
    meta.data.append(item)
    store.save_meta(model, version, operator, meta)
    return jsonify(meta.to_dict()), 201


@bp.route("/api/golden/<model>/<version>/<operator>/data-item/<int:idx>", methods=["DELETE"])
def delete_data_item(model, version, operator, idx):
    """Remove a data item by index."""
    store = _store()
    meta = store.load_meta(model, version, operator)
    if idx < 0 or idx >= len(meta.data):
        return jsonify({"error": "index out of range"}), 400
    meta.data.pop(idx)
    store.save_meta(model, version, operator, meta)
    return jsonify(meta.to_dict())


# -- create operator --------------------------------------------------------

@bp.route("/api/golden/create", methods=["POST"])
def create_golden():
    """Create operator with data items. Optionally upload files."""
    logger.info("[%s] create_golden start", _client_ip())
    model = request.form.get("model", "").strip()
    version = request.form.get("version", "").strip()
    operator = request.form.get("operator", "").strip()
    if not model or not version or not operator:
        return jsonify({"error": "model, version, operator are required"}), 400

    # Parse data items from form (JSON array in 'data_items' field)
    data_items: list[DataItem] = []
    raw_items = request.form.get("data_items", "")
    if raw_items:
        try:
            for d in json.loads(raw_items):
                item = DataItem.from_dict(d)
                data_items.append(item)
        except (json.JSONDecodeError, TypeError):
            return jsonify({"error": "data_items must be a JSON array"}), 400

    # 检查 data_items 内部 category+seq 重复
    seen = set()
    for item in data_items:
        key = (item.category, item.seq)
        if key in seen:
            return jsonify({"error": f"序号重复: {item.category}{item.seq}"}), 409
        seen.add(key)

    store = _store()
    # 算子名在同一模型/版本下不能重复
    existing_ops = store.list_operators(model, version)
    if operator in existing_ops:
        return jsonify({"error": f"算子「{operator}」已存在于 {model}/{version} 中，请使用不同名称"}), 409

    meta = OperatorMeta(name=operator, data=data_items)
    store.save_meta(model, version, operator, meta)
    store.log("create_operator", model=model, version=version, operator=operator,
              data_items=len(data_items), ip=_client_ip())

    saved_files = []
    files = request.files.getlist("file")
    for f in files:
        if f and f.filename and GoldenStore.allowed(f.filename):
            dest = store.save_file(model, version, operator, f.filename)
            f.save(str(dest))
            saved_files.append(f.filename)

    return jsonify({
        "model": model,
        "version": version,
        "operator": operator,
        "meta": meta.to_dict(),
        "files": saved_files,
    }), 201


# -- upload files -----------------------------------------------------------

@bp.route("/api/golden/upload", methods=["POST"])
def upload_golden():
    logger.info("[%s] upload_golden start", _client_ip())
    model = request.form.get("model", "").strip()
    version = request.form.get("version", "").strip()
    operator = request.form.get("operator", "").strip()

    f_single = request.files.get("file")
    if f_single and f_single.filename and (not model or not version):
        parsed = GoldenStore.parse_package_name(f_single.filename)
        if parsed:
            model = model or parsed[0]
            version = version or parsed[1]

    if not model or not version or not operator:
        return jsonify({"error": "model, version, operator are required"}), 400

    store = _store()
    saved = []
    files = request.files.getlist("file")
    for f in files:
        if not f or not f.filename:
            continue
        if not GoldenStore.allowed(f.filename):
            exts = ", ".join(ALLOWED_EXT)
            return jsonify({"error": f"only {exts} files accepted"}), 400
        dest = store.save_file(model, version, operator, f.filename)
        f.save(str(dest))
        saved.append(f.filename)

    if not saved:
        return jsonify({"error": "no valid files provided"}), 400

    # Auto-create data items from parsed filenames
    auto_items = []
    for fname in saved:
        parsed = parse_golden_filename(fname)
        if parsed:
            auto_items.append(parsed)

    if auto_items:
        meta = store.load_meta(model, version, operator)
        existing_seqs = {(d.category, d.seq) for d in meta.data}
        for p in auto_items:
            key = (p["category"], p["seq"])
            if key not in existing_seqs:
                meta.data.append(DataItem(
                    seq=p["seq"], category=p["category"],
                    loop=p["loop"], m=p["m"], n=p["n"], k=p["k"],
                    layout=p["layout"], precision=p["precision"],
                ))
                existing_seqs.add(key)
        store.save_meta(model, version, operator, meta)

    if saved:
        store.log("upload_files", model=model, version=version, operator=operator,
                  files=saved, ip=_client_ip())

    return jsonify({
        "model": model, "version": version, "operator": operator,
        "files": saved, "auto_parsed": len(auto_items),
    }), 201


# -- download ---------------------------------------------------------------

@bp.route("/api/golden/<model>/<version>/<operator>/download", methods=["GET"])
def download_operator(model, version, operator):
    store = _store()
    if operator not in store.list_operators(model, version):
        return jsonify({"error": "not found"}), 404
    buf = store.export_operator(model, version, operator)
    name = f"{model}_{version}_{operator}.zip"
    return send_file(buf, as_attachment=True, download_name=name,
                     mimetype="application/zip")


@bp.route("/api/golden/<model>/<version>/download", methods=["GET"])
def download_version(model, version):
    store = _store()
    if not store.list_operators(model, version):
        return jsonify({"error": "not found"}), 404
    buf = store.export_version(model, version)
    name = f"{model}_{version}.zip"
    return send_file(buf, as_attachment=True, download_name=name,
                     mimetype="application/zip")


@bp.route("/api/golden/<model>/download", methods=["GET"])
def download_model(model):
    store = _store()
    if not store.list_versions(model):
        return jsonify({"error": "not found"}), 404
    buf = store.export_model(model)
    name = f"{model}.zip"
    return send_file(buf, as_attachment=True, download_name=name,
                     mimetype="application/zip")


@bp.route("/api/golden/<model>/<version>/<operator>/<filename>/download", methods=["GET"])
def download_file(model, version, operator, filename):
    fp = _store().get_file(model, version, operator, filename)
    if not fp:
        return jsonify({"error": "not found"}), 404
    return send_from_directory(str(fp.parent), fp.name, as_attachment=True)


@bp.route("/api/golden/<model>/<version>/<operator>/<filename>/preview", methods=["GET"])
def preview_file(model, version, operator, filename):
    """Preview file content as hex (1 byte per line, up to 4KB)."""
    max_bytes = min(int(request.args.get("max", 4096)), 65536)
    store = _store()
    content = store.preview_file(model, version, operator, filename, max_bytes)
    if content is None:
        return jsonify({"error": "not found"}), 404
    fp = store.get_file(model, version, operator, filename)
    return jsonify({
        "filename": filename,
        "total_size": fp.stat().st_size if fp else 0,
        "preview_bytes": max_bytes,
        "hex": content,
    })


@bp.route("/api/golden/download-all", methods=["GET"])
def download_all_golden():
    store = _store()
    if not store.list():
        return jsonify({"error": "no data"}), 404
    buf = store.export_all()
    return send_file(buf, as_attachment=True, download_name="golden-all.zip",
                     mimetype="application/zip")


# -- fingerprint ------------------------------------------------------------

@bp.route("/api/golden/<model>/<version>/fingerprint", methods=["GET"])
def version_fingerprint(model, version):
    store = _store()
    fps = store.version_fingerprint(model, version)
    return jsonify({"model": model, "version": version, "operators": fps})


# -- import / migrate -------------------------------------------------------

@bp.route("/api/golden/import", methods=["POST"])
def import_golden():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "file is required"}), 400
    if not f.filename.endswith(".zip"):
        return jsonify({"error": "only .zip archives accepted"}), 400
    buf = io.BytesIO(f.read())
    imported = _store().import_archive(buf)
    return jsonify({"imported": imported, "count": len(imported)}), 201


@bp.route("/api/golden/sync", methods=["POST"])
def sync_golden():
    import urllib.error
    import urllib.request

    data = request.get_json(silent=True) or {}
    source = data.get("source", "").strip().rstrip("/")
    if not source:
        cfg = current_app.config.get("AITF_CONFIG")
        if cfg and cfg.server_url:
            source = cfg.server_url
    if not source:
        return jsonify({"error": "请提供同步地址或在 config.yaml 中配置 server"}), 400

    try:
        url = f"{source}/api/golden/download-all"
        resp = urllib.request.urlopen(url, timeout=300)
        buf = io.BytesIO(resp.read())
    except urllib.error.URLError as exc:
        return jsonify({"error": f"从服务器获取失败: {exc}"}), 502

    imported = _store().import_archive(buf)
    return jsonify({"source": source, "imported": imported, "count": len(imported)})


# -- operation log ----------------------------------------------------------

@bp.route("/api/golden/log", methods=["GET"])
def get_log():
    limit = min(int(request.args.get("limit", 100)), 500)
    return jsonify(_store().read_log(limit))


# -- layout conversion ------------------------------------------------------

def _lc():
    """Lazy-load layout_convert module."""
    from aitf.ds import layout_convert
    return layout_convert


@bp.route("/api/golden/convert-layout", methods=["POST"])
def convert_layout_file():
    """Convert a binary file's layout and/or dtype.

    Expects multipart: file, src_layout, dst_layout, src_precision, dst_precision, shape (JSON).
    Returns the converted file as download.
    """
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "file is required"}), 400

    src_layout = request.form.get("src_layout", "").strip()
    dst_layout = request.form.get("dst_layout", "").strip()
    src_prec = request.form.get("precision", request.form.get("src_precision", "FP16")).strip()
    dst_prec = request.form.get("dst_precision", src_prec).strip()

    if not src_layout or not dst_layout:
        return jsonify({"error": "src_layout and dst_layout are required"}), 400

    shape_str = request.form.get("shape", "{}")
    try:
        shape = json.loads(shape_str)
    except json.JSONDecodeError:
        return jsonify({"error": "shape must be valid JSON"}), 400

    lc = _lc()
    data = f.read()
    try:
        result = lc.convert(data, src_layout, dst_layout, src_prec, dst_prec, shape)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("[%s] convert failed: %s", _client_ip(), exc, exc_info=True)
        return jsonify({"error": f"转换失败: {exc}"}), 500

    # Replace layout/precision in filename if parseable
    out_name = f.filename
    parsed = parse_golden_filename(f.filename)
    if parsed:
        ext = "." + f.filename.rsplit(".", 1)[-1] if "." in f.filename else ".bin"
        item = DataItem(seq=parsed["seq"], category=parsed["category"],
                        loop=parsed["loop"], m=parsed["m"], n=parsed["n"],
                        k=parsed["k"], layout=dst_layout, precision=dst_prec)
        out_name = build_golden_filename(parsed["operator"], item, ext)

    buf = io.BytesIO(result)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=out_name,
                     mimetype="application/octet-stream")


@bp.route("/api/golden/conversions", methods=["GET"])
def list_conversions():
    """List supported layout and dtype conversion paths."""
    lc = _lc()
    return jsonify({
        "layout": [{"src": s, "dst": d} for s, d in lc.CONVERTERS.keys()],
        "dtype": "all-to-all",
    })


@bp.route("/api/golden/convert-txt", methods=["POST"])
def convert_txt_file():
    """Unified txt (hex) conversion: layout and/or dtype, jointly.

    Input/output are both txt format (1 byte per line, hex, no 0x).
    Form fields: file, src_layout, dst_layout, src_precision, dst_precision, shape (JSON).
    For blocked layouts (NC1HWC0, FRACTAL_NZ), layout+dtype are coupled and
    handled atomically by convert().
    """
    lc = _lc()
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "file is required"}), 400

    src_layout = request.form.get("src_layout", "").strip()
    dst_layout = request.form.get("dst_layout", "").strip()
    src_prec = request.form.get("src_precision", "").strip() or "FP16"
    dst_prec = request.form.get("dst_precision", "").strip() or src_prec

    # Default: if no dst_layout given, keep same layout
    if not src_layout:
        src_layout = "ND"
    if not dst_layout:
        dst_layout = src_layout

    shape_str = request.form.get("shape", "{}")
    try:
        shape = json.loads(shape_str)
    except json.JSONDecodeError:
        return jsonify({"error": "shape must be valid JSON"}), 400

    # Read txt -> bin
    text = f.read().decode("utf-8", errors="replace")
    data = lc.hex_txt_to_bin(text)

    # Unified conversion (layout + dtype jointly, with validation)
    try:
        data = lc.convert(data, src_layout, dst_layout, src_prec, dst_prec, shape)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("[%s] convert failed: %s", _client_ip(), exc, exc_info=True)
        return jsonify({"error": f"转换失败: {exc}"}), 500

    # bin -> txt
    result_txt = lc.bin_to_hex_txt(data)

    # Build output filename
    out_name = f.filename
    if out_name.endswith(".txt"):
        out_name = out_name[:-4]
    if src_layout != dst_layout:
        out_name += f"_{dst_layout}"
    if src_prec != dst_prec:
        out_name += f"_{dst_prec}"
    out_name += ".txt"

    buf = io.BytesIO(result_txt.encode("utf-8"))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=out_name,
                     mimetype="text/plain")


@bp.route("/api/golden/<model>/<version>/<operator>/convert", methods=["POST"])
def convert_operator_file(model, version, operator):
    """Convert an existing file: layout and/or dtype (jointly).

    JSON body: { filename, dst_layout, dst_precision }
    Uses existing data item metadata for shape info.
    """
    data = request.get_json(silent=True) or {}
    filename = data.get("filename", "")
    dst_layout = data.get("dst_layout", "")
    dst_prec = data.get("dst_precision", "")
    if not filename or (not dst_layout and not dst_prec):
        return jsonify({"error": "filename and dst_layout/dst_precision are required"}), 400

    store = _store()
    fp = store.get_file(model, version, operator, filename)
    if not fp:
        return jsonify({"error": "file not found"}), 404

    parsed = parse_golden_filename(filename)
    if not parsed:
        return jsonify({"error": "无法解析文件名以确定源格式"}), 400

    lc = _lc()
    src_layout = parsed["layout"]
    src_prec = parsed["precision"]
    dst_layout = dst_layout or src_layout
    dst_prec = dst_prec or src_prec

    if src_layout == dst_layout and src_prec == dst_prec:
        return jsonify({"error": "源格式与目标格式完全相同"}), 400
    if src_layout != dst_layout and not lc.can_convert(src_layout, dst_layout):
        return jsonify({"error": f"不支持的分型转换: {src_layout} -> {dst_layout}"}), 400

    raw = fp.read_bytes()
    shape = {"loop": parsed["loop"], "m": parsed["m"],
             "n": parsed["n"], "k": parsed["k"]}

    try:
        result = lc.convert(raw, src_layout, dst_layout, src_prec, dst_prec, shape)
    except Exception as exc:
        logger.error("[%s] convert failed: %s", _client_ip(), exc, exc_info=True)
        return jsonify({"error": f"转换失败: {exc}"}), 500

    # Build new filename
    item = DataItem(seq=parsed["seq"], category=parsed["category"],
                    loop=parsed["loop"], m=parsed["m"], n=parsed["n"],
                    k=parsed["k"], layout=dst_layout, precision=dst_prec)
    ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ".bin"
    new_name = build_golden_filename(operator, item, ext)

    dest = store.save_file(model, version, operator, new_name)
    dest.write_bytes(result)

    # Update data item layout+precision in meta
    meta = store.load_meta(model, version, operator)
    for d in meta.data:
        if d.category == parsed["category"] and d.seq == parsed["seq"]:
            d.layout = dst_layout
            d.precision = dst_prec
            break
    store.save_meta(model, version, operator, meta)
    store.log("convert", model=model, version=version, operator=operator,
              filename=filename, src_layout=src_layout, dst_layout=dst_layout,
              src_prec=src_prec, dst_prec=dst_prec, new_file=new_name, ip=_client_ip())

    return jsonify({
        "original": filename,
        "converted": new_name,
        "src_layout": src_layout,
        "dst_layout": dst_layout,
    }), 201


# -- pth inspect/export (无需 torch 依赖，纯 pickle 解析) -------------------

def _pth_parser():
    from aitf.ds import pth_parser
    return pth_parser


@bp.route("/api/golden/pth-inspect", methods=["POST"])
def pth_inspect():
    """解析 .pth 文件，列出内部所有 tensor 的 key/dtype/shape。

    使用自定义 pickle unpickler 解析，无需 torch 依赖。
    注意: pth 文件可能借用了 torch 序列化格式，实际存储 Ascend 自定义格式数据。
    """
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "file is required"}), 400
    if not f.filename.endswith(".pth"):
        return jsonify({"error": "仅支持 .pth 文件"}), 400

    try:
        pp = _pth_parser()
        tensors = pp.inspect_pth(f.stream)
        return jsonify({"tensors": tensors, "count": len(tensors)})
    except Exception as exc:
        logger.error("[%s] pth inspect failed: %s", _client_ip(), exc, exc_info=True)
        return jsonify({"error": f"pth 解析失败: {exc}"}), 500


@bp.route("/api/golden/pth-export", methods=["POST"])
def pth_export():
    """从 .pth 文件中导出指定 tensor 为 txt 格式（每行1字节hex，无0x）。"""
    f = request.files.get("file")
    key = request.form.get("key", "").strip()
    if not f or not f.filename:
        return jsonify({"error": "file is required"}), 400
    if not key:
        return jsonify({"error": "key is required"}), 400

    try:
        pp = _pth_parser()
        lc = _lc()
        raw, info = pp.export_tensor(f.stream, key)

        # 原样导出 raw bytes 为 txt (不做精度/分型转换)
        txt = lc.bin_to_hex_txt(raw)
        safe_name = info.get("operator", key.replace(".", "_")) + "_" + info.get("category", "other") + ".txt"
        buf = io.BytesIO(txt.encode("utf-8"))
        buf.seek(0)

        logger.info("[%s] pth export: %s (%s %s, %d bytes)",
                    _client_ip(), key, info["dtype"], info["shape"], len(raw))

        return send_file(buf, as_attachment=True, download_name=safe_name,
                         mimetype="text/plain")
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("[%s] pth export failed: %s", _client_ip(), exc, exc_info=True)
        return jsonify({"error": f"pth 导出失败: {exc}"}), 500


# -- dbox export ------------------------------------------------------------

@bp.route("/api/golden/<model>/<version>/export-dbox", methods=["POST"])
def export_to_dbox(model, version):
    """Export version data to DBox server.

    Expects JSON body with optional ``dbox_url`` (defaults to app config
    ``DBOX_SERVER``).  Packages the version as a zip and uploads it via
    the DBox upload API.
    """
    store = _store()
    if not store.list_operators(model, version):
        return jsonify({"error": "not found"}), 404

    data = request.get_json(silent=True) or {}
    dbox_url = (data.get("dbox_url", "") or
                current_app.config.get("DBOX_SERVER", "")).strip().rstrip("/")
    if not dbox_url:
        return jsonify({"error": "DBox 服务器地址未配置，请在设置中配置 DBOX_SERVER 或请求中提供 dbox_url"}), 400

    buf = store.export_version(model, version)
    filename = f"{model}_{version}.zip"

    upload_path = current_app.config.get("DBOX_UPLOAD_PATH", "/api/upload")
    # DBox API: POST <dbox_url><upload_path>  multipart/form-data
    # 字段: file (zip), model (模型名), version (版本号)
    # 算子列表通过 operators JSON 字段传递
    try:
    
        import urllib.request

        operators = store.list_operators(model, version)
        file_data = buf.read()
        boundary = "----AitfDboxBoundary"

        def _field(name: str, value: str) -> bytes:
            return (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode()

        def _file_field(name: str, fname: str, data: bytes, content_type: str) -> bytes:
            return (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{fname}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode() + data + b"\r\n"

        body = (
            _field("model", model)
            + _field("version", version)
            + _field("operators", json.dumps(operators))
            + _file_field("file", filename, file_data, "application/zip")
            + f"--{boundary}--\r\n".encode()
        )

        req = urllib.request.Request(
            f"{dbox_url}{upload_path}",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=300)
        result = resp.read().decode()
    except Exception as exc:
        logger.error("[%s] DBox upload failed: %s", _client_ip(), exc, exc_info=True)
        return jsonify({"error": f"上传到 DBox 失败: {exc}"}), 502

    return jsonify({
        "model": model,
        "version": version,
        "dbox_url": dbox_url,
        "filename": filename,
        "result": result,
    })


# -- delete -----------------------------------------------------------------

@bp.route("/api/golden/<model>/<version>/<operator>", methods=["DELETE"])
def delete_operator(model, version, operator):
    store = _store()
    if not store.delete_operator(model, version, operator):
        return jsonify({"error": "not found"}), 404
    store.log("delete_operator", model=model, version=version, operator=operator, ip=_client_ip())
    return jsonify({"deleted": f"{model}/{version}/{operator}"})


@bp.route("/api/golden/<model>/<version>", methods=["DELETE"])
def delete_version(model, version):
    store = _store()
    if not store.delete_version(model, version):
        return jsonify({"error": "not found"}), 404
    store.log("delete_version", model=model, version=version, ip=_client_ip())
    return jsonify({"deleted": f"{model}/{version}"})


@bp.route("/api/golden/<model>", methods=["DELETE"])
def delete_model(model):
    store = _store()
    if not store.delete_model(model):
        return jsonify({"error": "not found"}), 404
    store.log("delete_model", model=model, ip=_client_ip())
    return jsonify({"deleted": model})


@bp.route("/api/golden/<model>/<version>/copy-from/<source_version>", methods=["POST"])
def copy_version_structure(model, version, source_version):
    """Copy operator structure (metadata only) from source_version to version."""
    store = _store()
    if not store.list_operators(model, source_version):
        return jsonify({"error": f"源版本 {source_version} 不存在或无算子"}), 404
    copied = store.copy_version_structure(model, source_version, version)
    return jsonify({"copied": copied, "count": len(copied)}), 201


@bp.route("/api/golden/<model>/<version>/clear", methods=["DELETE"])
def clear_version(model, version):
    """Delete all operators under a version."""
    store = _store()
    count = store.clear_version(model, version)
    if count == 0:
        return jsonify({"error": "no operators to clear"}), 404
    return jsonify({"cleared": count})


@bp.route("/api/golden/<model>/<version>/<operator>/<filename>", methods=["DELETE"])
def delete_file(model, version, operator, filename):
    if not _store().delete_file(model, version, operator, filename):
        return jsonify({"error": "not found"}), 404
    return jsonify({"deleted": f"{model}/{version}/{operator}/{filename}"})
