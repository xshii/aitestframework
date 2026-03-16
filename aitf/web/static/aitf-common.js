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
