"""REST API routes for golden data management.

Model → Version → Operator → Data Items (category/shape/layout/precision).
"""

from __future__ import annotations

import io

from flask import Blueprint, current_app, jsonify, request, send_file, send_from_directory

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

bp = Blueprint("datastore", __name__, template_folder="templates")


def _store() -> GoldenStore:
    base = current_app.config.get("DATASTORE_BASE_DIR", "datastore")
    return GoldenStore(base)


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


@bp.route("/api/golden/choices", methods=["GET"])
def get_choices():
    return jsonify({
        "category": list(CATEGORY_CHOICES),
        "precision": list(PRECISION_CHOICES),
        "layout": list(LAYOUT_CHOICES),
        "filename_format": "{op}_{category}{seq}_{layout}_{precision}_{shape}.ext",
        "filename_example": "matmul_input0_NCHW_fp16_128x96.bin",
        "allowed_ext": list(ALLOWED_EXT),
    })


@bp.route("/api/golden/parse-filename", methods=["POST"])
def parse_filename():
    """Parse a golden filename and return extracted metadata."""
    data = request.get_json(silent=True) or {}
    filename = data.get("filename", "")
    result = parse_golden_filename(filename)
    if not result:
        return jsonify({"error": "无法解析文件名，格式应为: {op}_{category}{seq}_{layout}_{precision}_{shape}.ext"}), 400
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
    _store().save_meta(model, version, operator, meta)
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
    model = request.form.get("model", "").strip()
    version = request.form.get("version", "").strip()
    operator = request.form.get("operator", "").strip()
    if not model or not version or not operator:
        return jsonify({"error": "model, version, operator are required"}), 400

    # Parse data items from form (JSON array in 'data_items' field)
    data_items: list[DataItem] = []
    raw_items = request.form.get("data_items", "")
    if raw_items:
        import json
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


@bp.route("/api/golden/<model>/<version>/<operator>/<filename>/download", methods=["GET"])
def download_file(model, version, operator, filename):
    fp = _store().get_file(model, version, operator, filename)
    if not fp:
        return jsonify({"error": "not found"}), 404
    return send_from_directory(str(fp.parent), fp.name, as_attachment=True)


@bp.route("/api/golden/download-all", methods=["GET"])
def download_all_golden():
    store = _store()
    if not store.list():
        return jsonify({"error": "no data"}), 404
    buf = store.export_all()
    return send_file(buf, as_attachment=True, download_name="golden-all.zip",
                     mimetype="application/zip")


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
        return jsonify({"error": "source URL is required"}), 400

    try:
        url = f"{source}/api/golden/download-all"
        resp = urllib.request.urlopen(url, timeout=300)
        buf = io.BytesIO(resp.read())
    except urllib.error.URLError as exc:
        return jsonify({"error": f"failed to fetch from {source}: {exc}"}), 502

    imported = _store().import_archive(buf)
    return jsonify({"source": source, "imported": imported, "count": len(imported)})


# -- delete -----------------------------------------------------------------

@bp.route("/api/golden/<model>/<version>/<operator>", methods=["DELETE"])
def delete_operator(model, version, operator):
    if not _store().delete_operator(model, version, operator):
        return jsonify({"error": "not found"}), 404
    return jsonify({"deleted": f"{model}/{version}/{operator}"})


@bp.route("/api/golden/<model>/<version>", methods=["DELETE"])
def delete_version(model, version):
    if not _store().delete_version(model, version):
        return jsonify({"error": "not found"}), 404
    return jsonify({"deleted": f"{model}/{version}"})


@bp.route("/api/golden/<model>", methods=["DELETE"])
def delete_model(model):
    if not _store().delete_model(model):
        return jsonify({"error": "not found"}), 404
    return jsonify({"deleted": model})


@bp.route("/api/golden/<model>/<version>/<operator>/<filename>", methods=["DELETE"])
def delete_file(model, version, operator, filename):
    if not _store().delete_file(model, version, operator, filename):
        return jsonify({"error": "not found"}), 404
    return jsonify({"deleted": f"{model}/{version}/{operator}/{filename}"})
