"""Single-page index combining Data, Deps and Logs tabs."""

from __future__ import annotations

from flask import Blueprint, render_template

from aitf.web.views import build_log_listing, log_root

index_bp = Blueprint("index", __name__)


@index_bp.route("/")
def index():
    return render_template("index.html", log_entries=build_log_listing(log_root()))
