"""Tiny Flask admin UI for the face DB and projects board.

Routes:

  /                                HTML (employees + projects + metrics)
  /api/employees                   list/create/...
  /api/employees/<id>              GET / DELETE / rename
  /api/projects                    GET / POST (create)
  /api/projects/<id>               GET / POST edit / POST delete
  /api/metrics                     {interactions_total: N}

Same SQLite file as main.py. SQLite multi-reader / single-writer keeps
concurrency safe for kiosk-level traffic.
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
<title>Echo AI admin</title>
<style>
 body { font-family: system-ui, sans-serif; max-width: 980px; margin: 32px auto; padding: 0 16px; color: #1a1a1a; }
 h1 { margin-bottom: 0; }
 h2 { margin-top: 32px; }
 .sub { color: #666; margin-top: 4px; font-size: 13px; }
 table { border-collapse: collapse; width: 100%; margin-top: 12px; }
 th, td { padding: 8px 12px; border-bottom: 1px solid #ddd; text-align: left; vertical-align: top; }
 th { background: #f5f5f5; }
 .actions button { margin-right: 6px; }
 .empty { color: #888; padding: 24px; text-align: center; }
 input[type=text], textarea { padding: 4px 8px; font-size: 14px; width: 100%; box-sizing: border-box; }
 textarea { min-height: 60px; }
 form.row td { padding: 4px 6px; }
 .badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
          background: #eef; color: #335; font-size: 12px; }
 .metric { display: inline-block; padding: 8px 14px; background: #f7f7f7; border-radius: 8px; margin-right: 8px; }
 .metric .v { font-size: 24px; font-weight: bold; }
 button { cursor: pointer; padding: 4px 12px; }
 .danger { color: #b00; border-color: #b00; }
</style>
</head>
<body>
<h1>Echo AI admin</h1>
<p class="sub">DB: {{ db_path }}</p>

<p>
 <span class="metric">interactions <span class="v">{{ interactions_total }}</span></span>
 <span class="metric">employees <span class="v">{{ emp_rows|length }}</span></span>
 <span class="metric">projects <span class="v">{{ proj_rows|length }}</span></span>
</p>

<h2>Projects</h2>
{% if proj_rows %}
<table>
 <thead><tr>
  <th>#</th><th>Title</th><th>Description</th><th>Order</th><th>Created</th><th>Actions</th>
 </tr></thead>
 <tbody>
 {% for pid, title, desc, ordering, created in proj_rows %}
  <tr id="p-{{ pid }}">
   <td><code>{{ pid }}</code></td>
   <td><span class="title">{{ title }}</span></td>
   <td><span class="desc">{{ desc or "" }}</span></td>
   <td>{{ ordering }}</td>
   <td>{{ created }}</td>
   <td class="actions">
    <button onclick="editProject({{ pid }})">Edit</button>
    <button class="danger" onclick="deleteProject({{ pid }})">Delete</button>
   </td>
  </tr>
 {% endfor %}
 </tbody>
</table>
{% else %}
<p class="empty">No projects yet. Add one below.</p>
{% endif %}

<form id="add-project" onsubmit="addProject(event)" style="margin-top:16px">
 <table><tr>
  <td style="width:30%"><input type="text" name="title" placeholder="Title" required></td>
  <td style="width:55%"><input type="text" name="description" placeholder="Description"></td>
  <td style="width:8%"><input type="text" name="ordering" placeholder="Order" value="0"></td>
  <td><button type="submit">Add</button></td>
 </tr></table>
</form>

<h2>Employees</h2>
{% if emp_rows %}
<table>
 <thead><tr>
  <th>ID</th><th>Name</th><th>Samples</th><th>Created</th><th>Actions</th>
 </tr></thead>
 <tbody>
 {% for emp_id, name, count, created in emp_rows %}
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
<p class="empty">No employees registered yet.</p>
{% endif %}

<script>
async function rename_(id) {
  const cur = document.querySelector(`#row-${id} .name`).textContent;
  const next = prompt(`New name for ${id}:`, cur);
  if (!next || next === cur) return;
  const r = await fetch(`/api/employees/${id}/rename`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name: next}),
  });
  if (r.ok) location.reload(); else alert('rename failed: ' + (await r.text()));
}
async function del(id) {
  if (!confirm(`Delete ${id} and all face samples?`)) return;
  const r = await fetch(`/api/employees/${id}/delete`, {method: 'POST'});
  if (r.ok) location.reload(); else alert('delete failed: ' + (await r.text()));
}
async function addProject(ev) {
  ev.preventDefault();
  const f = ev.target;
  const body = {
    title: f.title.value, description: f.description.value,
    ordering: parseInt(f.ordering.value || '0', 10),
  };
  const r = await fetch('/api/projects', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  if (r.ok) location.reload(); else alert('add failed: ' + (await r.text()));
}
async function editProject(pid) {
  const row = document.getElementById(`p-${pid}`);
  const cur = {
    title: row.querySelector('.title').textContent,
    description: row.querySelector('.desc').textContent,
  };
  const title = prompt('Title:', cur.title);
  if (title === null) return;
  const desc = prompt('Description:', cur.description);
  if (desc === null) return;
  const r = await fetch(`/api/projects/${pid}/edit`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({title, description: desc}),
  });
  if (r.ok) location.reload(); else alert('edit failed: ' + (await r.text()));
}
async function deleteProject(pid) {
  if (!confirm(`Delete project ${pid}?`)) return;
  const r = await fetch(`/api/projects/${pid}/delete`, {method: 'POST'});
  if (r.ok) location.reload(); else alert('delete failed: ' + (await r.text()));
}
</script>
</body>
</html>
"""


def _db():
    return FaceDB()


# ---------------- HTML ----------------


@app.route("/")
def index():
    db = _db()
    try:
        emp_rows = db.list_employees()
        proj_rows = db.list_projects()
        interactions_total = db.interaction_count_total()
    finally:
        db.close()
    from config import DB_PATH
    return render_template_string(
        INDEX_HTML,
        emp_rows=emp_rows,
        proj_rows=proj_rows,
        interactions_total=interactions_total,
        db_path=str(DB_PATH),
    )


# ---------------- employees ----------------


@app.route("/api/employees")
def api_emp_list():
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
def api_emp_show(emp_id):
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
def api_emp_rename(emp_id):
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
def api_emp_delete(emp_id):
    db = _db()
    try:
        if not db.delete_employee(emp_id):
            return ("not found", 404)
    finally:
        db.close()
    return jsonify({"ok": True, "emp_id": emp_id})


# ---------------- projects ----------------


@app.route("/api/projects")
def api_proj_list():
    db = _db()
    try:
        rows = db.list_projects()
    finally:
        db.close()
    return jsonify([
        {"id": r[0], "title": r[1], "description": r[2],
         "ordering": r[3], "created_at": r[4]}
        for r in rows
    ])


@app.route("/api/projects", methods=["POST"])
def api_proj_create():
    body = request.get_json(silent=True) or {}
    title = (body.get("title") or "").strip()
    if not title:
        return ("title required", 400)
    description = (body.get("description") or "").strip()
    try:
        ordering = int(body.get("ordering", 0))
    except (TypeError, ValueError):
        ordering = 0
    db = _db()
    try:
        pid = db.add_project(title, description, ordering)
    finally:
        db.close()
    return jsonify({"ok": True, "id": pid})


@app.route("/api/projects/<int:pid>")
def api_proj_show(pid):
    db = _db()
    try:
        row = db.get_project(pid)
    finally:
        db.close()
    if not row:
        return ("not found", 404)
    return jsonify({"id": row[0], "title": row[1], "description": row[2],
                    "ordering": row[3], "created_at": row[4]})


@app.route("/api/projects/<int:pid>/edit", methods=["POST"])
def api_proj_edit(pid):
    body = request.get_json(silent=True) or {}
    kwargs = {}
    if "title" in body:
        kwargs["title"] = (body["title"] or "").strip()
    if "description" in body:
        kwargs["description"] = (body["description"] or "").strip()
    if "ordering" in body:
        try:
            kwargs["ordering"] = int(body["ordering"])
        except (TypeError, ValueError):
            pass
    if not kwargs:
        return ("no fields to update", 400)
    db = _db()
    try:
        if not db.update_project(pid, **kwargs):
            return ("not found", 404)
    finally:
        db.close()
    return jsonify({"ok": True})


@app.route("/api/projects/<int:pid>/delete", methods=["POST"])
def api_proj_delete(pid):
    db = _db()
    try:
        if not db.delete_project(pid):
            return ("not found", 404)
    finally:
        db.close()
    return jsonify({"ok": True})


# ---------------- metrics ----------------


@app.route("/api/metrics")
def api_metrics():
    db = _db()
    try:
        return jsonify({"interactions_total": db.interaction_count_total()})
    finally:
        db.close()


def main():
    host = os.environ.get("ADMIN_HOST", "0.0.0.0")
    port = int(os.environ.get("ADMIN_PORT", "8081"))
    print(f"echo admin UI on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
