/* aitf-common.js — shared utility functions for all tab templates. */

function _esc(s) {
  return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function _fmtTime(ts) {
  if (!ts) return '-';
  /* Auto-detect: epoch seconds (number < 1e12) vs ISO string vs epoch ms */
  var d;
  if (typeof ts === 'number') {
    d = new Date(ts < 1e12 ? ts * 1000 : ts);
  } else {
    d = new Date(ts);
    if (isNaN(d.getTime())) d = new Date(ts + 'Z');
  }
  if (isNaN(d.getTime())) return '-';
  var p = function(v) { return v < 10 ? '0' + v : v; };
  return d.getFullYear() + '-' + p(d.getMonth() + 1) + '-' + p(d.getDate())
    + ' ' + p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds());
}

function _pct(v) { return (v * 100).toFixed(1) + '%'; }

function _fmtSize(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}

/**
 * _initTableDrag — enable drag-and-drop row reordering on a <tbody>.
 *
 * @param {string}   tbodyId   id of the <tbody> element
 * @param {function} onReorder (tbody) => { … } called after a drop
 */
function _initTableDrag(tbodyId, onReorder) {
  var tb = document.getElementById(tbodyId);
  var dragRow = null;
  tb.querySelectorAll('tr[draggable]').forEach(function(tr) {
    tr.addEventListener('dragstart', function(e) { dragRow = tr; tr.style.opacity = '0.4'; e.dataTransfer.effectAllowed = 'move'; });
    tr.addEventListener('dragend', function() { tr.style.opacity = '1'; dragRow = null; });
    tr.addEventListener('dragover', function(e) { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; });
    tr.addEventListener('drop', function(e) {
      e.preventDefault();
      if (!dragRow || dragRow === tr) return;
      var rows = Array.from(tb.querySelectorAll('tr'));
      var fromIdx = rows.indexOf(dragRow), toIdx = rows.indexOf(tr);
      if (fromIdx < toIdx) tb.insertBefore(dragRow, tr.nextSibling);
      else tb.insertBefore(dragRow, tr);
      if (onReorder) onReorder(tb);
    });
  });
}

/**
 * _buildVerSelect — build a version <select> dropdown.
 *
 * @param {string[]} versions  available versions
 * @param {string}   current   currently selected version
 * @param {string}   [attrs]   extra attributes for <select> (e.g. 'id="x"')
 */
function _buildVerSelect(versions, current, attrs) {
  var sel = '<select class="form-select form-select-sm" style="width:auto;min-width:90px"' + (attrs ? ' ' + attrs : '') + '>';
  versions.forEach(function(v) {
    sel += '<option value="' + _esc(v) + '"' + (v === current ? ' selected' : '') + '>' + _esc(v) + '</option>';
  });
  return sel + '</select>';
}

/**
 * _renderExeRow — render a single execution as a <tr>.
 *
 * @param {object}   e         execution object
 * @param {object}   [opts]    { onClick: 'fnName', showNotRun: bool }
 */
function _renderExeRow(e, opts) {
  opts = opts || {};
  var rate = e.total ? _pct(e.pass_rate) : '\u2014';
  var idCell = opts.onClick
    ? '<a href="#" onclick="' + opts.onClick + '(\'' + _esc(e.id) + '\');return false;" class="text-decoration-none">' + _esc(e.id) + '</a>'
    : '<small>' + _esc(e.id) + '</small>';
  var html = '<tr>'
    + '<td>' + idCell + '</td>'
    + '<td>' + _fmtTime(e.started_at) + '</td>'
    + '<td>' + _esc(e.bundle || '-') + '</td>'
    + '<td>' + (e.total || 0) + '</td>'
    + '<td class="text-success">' + (e.passed || 0) + '</td>'
    + '<td class="text-danger">' + (e.failed_total || 0) + '</td>';
  if (opts.showNotRun) html += '<td class="text-muted">' + (e.not_run || 0) + '</td>';
  html += '<td>' + rate + '</td></tr>';
  return html;
}

/**
 * _syncAction — unified sync-from-server pattern.
 *
 * @param {string}   url       POST endpoint
 * @param {string}   msgId     id of the <span> for feedback text
 * @param {object}   [opts]    extra fetch options (headers, body, …)
 * @param {function} onOk      (data) => { reload…; return '成功描述'; }
 */
function _syncAction(url, msgId, opts, onOk) {
  var msg = document.getElementById(msgId);
  if (msg) msg.textContent = '同步中...';
  fetch(url, Object.assign({method: 'POST'}, opts || {}))
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.error) { if (msg) msg.textContent = '同步失败: ' + d.error; }
      else { if (msg) msg.textContent = onOk(d); setTimeout(function() { if (msg) msg.textContent = ''; }, 3000); }
    })
    .catch(function(e) { if (msg) msg.textContent = '同步失败: ' + e; });
}
