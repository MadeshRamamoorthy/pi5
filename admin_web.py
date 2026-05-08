"""Tiny Flask admin UI for the face DB.

  GET  /                  HTML table of employees + sample counts
  GET  /api/employees     JSON list
  GET  /api/employees/<id>  JSON details for one
  POST /api/employees/<id>/rename {name: ...}
  POST /api/employees/<id>/delete

Run:
    ./start_admin.sh        # binds to 0.0.0.0:8080 by default
    ADMIN_PORT=9000 ./start_admin.sh

The DB is the same SQLite file the main app uses. SQLite supports
multi-reader / single-writer concurrency, so this can run in parallel
with main.py without lock contention for typical kiosk traffic.
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template_string, request

from database import FaceDB

app = Flask(__name__)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>pi5 face DB</title>
<style>
 body { font-family: system-ui, sans-serif; max-width: 880px; margin: 32px auto; padding: 0 16px; }
 h1 { margin-bottom: 4px; }
 .sub { color: #666; margin-top: 0; font-size: 13px; }
 table { border-collapse: collapse; width: 100%; margin-top: 16px; }
 th, td { padding: 8px 12px; border-bottom: 1px solid #ddd; text-align: left; }
 th { background: #f5f5f5; }
 .actions button { margin-right: 6px; }
 .empty { color: #888; padding: 32px; text-align: center; }
 input[type=text] { padding: 4px 8px; font-size: 14px; }
 .badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
          background: #eef; color: #335; font-size: 12px; }
 button { cursor: pointer; padding: 4px 12px; }
 .danger { color: #b00; border-color: #b00; }
</style>
</head>
<body>
<h1>Employees</h1>
<p class="sub">DB: {{ db_path }} &middot; total: {{ rows|length }}</p>

{% if rows %}
<table>
 <thead><tr>
  <th>ID</th><th>Name</th><th>Samples</th><th>Created</th><th>Actions</th>
 </tr></thead>
 <tbody>
 {% for emp_id, name, count, created in rows %}
  <tr id="row-{{ emp_id }}">
   <td><code>{{ emp_id }}</code></td>
   <td><span class="name">{{ name }}</span></td>
   <td><span class="badge">{{ count }}</span></td>
   <td>{{ created }}</td>
   <td class="actions">
    <button onclick="rename_('{{ emp_id }}')">Rename</button>
    <button class="danger" onclick="del('{{ emp_id }}')">Delete</button>
   </td>
  </tr>
 {% endfor %}
 </tbody>
</table>
{% else %}
<p class="empty">No employees registered yet. Wake the system with "hello echo"
and let it auto-register a new face.</p>
{% endif %}

<script>
async function rename_(id) {
  const cur = document.querySelector(`#row-${id} .name`).textContent;
  const next = prompt(`New name for ${id}:`, cur);
  if (!next || next === cur) return;
  const r = await fetch(`/api/employees/${id}/rename`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name: next}),
  });
  if (r.ok) location.reload();
  else alert('rename failed: ' + (await r.text()));
}
async function del(id) {
  if (!confirm(`Delete ${id} and all face samples? This cannot be undone.`)) return;
  const r = await fetch(`/api/employees/${id}/delete`, {method: 'POST'});
  if (r.ok) location.reload();
  else alert('delete failed: ' + (await r.text()));
}
</script>
</body>
</html>
"""


def _db():
    return FaceDB()


@app.route("/")
def index():
    db = _db()
    try:
        rows = db.list_employees()
    finally:
        db.close()
    from config import DB_PATH
    return render_template_string(INDEX_HTML, rows=rows, db_path=str(DB_PATH))


@app.route("/api/employees")
def api_list():
    db = _db()
    try:
        rows = db.list_employees()
    finally:
        db.close()
    return jsonify([
        {"emp_id": r[0], "name": r[1], "samples": r[2], "created_at": r[3]}
        for r in rows
    ])


@app.route("/api/employees/<emp_id>")
def api_show(emp_id):
    db = _db()
    try:
        if not db.employee_exists(emp_id):
            return jsonify({"error": "not found"}), 404
        name = db.get_name(emp_id)
        count = db.count_embeddings(emp_id)
    finally:
        db.close()
    return jsonify({"emp_id": emp_id, "name": name, "samples": count})


@app.route("/api/employees/<emp_id>/rename", methods=["POST"])
def api_rename(emp_id):
    body = request.get_json(silent=True) or {}
    new_name = (body.get("name") or "").strip()
    if not new_name:
        return ("name required", 400)
    db = _db()
    try:
        if not db.employee_exists(emp_id):
            return ("not found", 404)
        db.conn.execute(
            "UPDATE employees SET name = ? WHERE emp_id = ?",
            (new_name, emp_id),
        )
        db.conn.commit()
    finally:
        db.close()
    return jsonify({"ok": True, "emp_id": emp_id, "name": new_name})


@app.route("/api/employees/<emp_id>/delete", methods=["POST"])
def api_delete(emp_id):
    db = _db()
    try:
        if not db.delete_employee(emp_id):
            return ("not found", 404)
    finally:
        db.close()
    return jsonify({"ok": True, "emp_id": emp_id})


def main():
    host = os.environ.get("ADMIN_HOST", "0.0.0.0")
    port = int(os.environ.get("ADMIN_PORT", "8080"))
    print(f"pi5 admin UI on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
