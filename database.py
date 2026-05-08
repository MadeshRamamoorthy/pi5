from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np

from config import DB_PATH, EMBEDDING_DIM

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    emp_id     TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    emp_id     TEXT NOT NULL,
    embedding  BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_face_emp ON face_embeddings(emp_id);

CREATE TABLE IF NOT EXISTS projects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    description TEXT,
    ordering    INTEGER DEFAULT 0,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS interactions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    emp_id     TEXT NOT NULL,
    session_id TEXT NOT NULL,
    ts         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(emp_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions(session_id);
"""


class FaceDB:
    def __init__(self, path: Path = DB_PATH):
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def add_employee(self, emp_id: str, name: str, embeddings: Iterable[np.ndarray]):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO employees (emp_id, name) VALUES (?, ?)",
            (emp_id, name),
        )
        for emb in embeddings:
            cur.execute(
                "INSERT INTO face_embeddings (emp_id, embedding) VALUES (?, ?)",
                (emp_id, _to_blob(emb)),
            )
        self.conn.commit()

    def add_embedding(self, emp_id: str, embedding: np.ndarray):
        self.conn.execute(
            "INSERT INTO face_embeddings (emp_id, embedding) VALUES (?, ?)",
            (emp_id, _to_blob(embedding)),
        )
        self.conn.commit()

    def load_all(self):
        """Return (emp_ids, names, matrix) where matrix is (N, D) float32."""
        rows = self.conn.execute(
            """
            SELECT f.emp_id, e.name, f.embedding
            FROM face_embeddings f
            JOIN employees e ON e.emp_id = f.emp_id
            """
        ).fetchall()
        if not rows:
            return [], [], np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        emp_ids = [r[0] for r in rows]
        names = [r[1] for r in rows]
        matrix = np.stack([_from_blob(r[2]) for r in rows]).astype(np.float32)
        return emp_ids, names, matrix

    def employee_exists(self, emp_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM employees WHERE emp_id = ?", (emp_id,)
        ).fetchone()
        return row is not None

    def get_name(self, emp_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT name FROM employees WHERE emp_id = ?", (emp_id,)
        ).fetchone()
        return row[0] if row else None

    def list_employees(self):
        """Return [(emp_id, name, sample_count, created_at)] sorted by emp_id."""
        return self.conn.execute(
            """
            SELECT e.emp_id, e.name, COUNT(f.id), e.created_at
            FROM employees e
            LEFT JOIN face_embeddings f ON f.emp_id = e.emp_id
            GROUP BY e.emp_id
            ORDER BY e.emp_id
            """
        ).fetchall()

    def delete_employee(self, emp_id: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM employees WHERE emp_id = ?", (emp_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def get_embeddings(self, emp_id: str) -> np.ndarray:
        rows = self.conn.execute(
            "SELECT embedding FROM face_embeddings WHERE emp_id = ?",
            (emp_id,),
        ).fetchall()
        if not rows:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        return np.stack([_from_blob(r[0]) for r in rows]).astype(np.float32)

    def count_embeddings(self, emp_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM face_embeddings WHERE emp_id = ?",
            (emp_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    def trim_embeddings(self, emp_id: str, keep_n: int) -> int:
        """Keep the newest `keep_n` embeddings for this emp_id, drop older
        ones. Returns the number deleted."""
        if keep_n <= 0:
            return 0
        rows = self.conn.execute(
            "SELECT id FROM face_embeddings WHERE emp_id = ? "
            "ORDER BY id DESC LIMIT -1 OFFSET ?",
            (emp_id, keep_n),
        ).fetchall()
        if not rows:
            return 0
        ids = [r[0] for r in rows]
        self.conn.executemany(
            "DELETE FROM face_embeddings WHERE id = ?",
            [(i,) for i in ids],
        )
        self.conn.commit()
        return len(ids)

    # ---- projects --------------------------------------------------------

    def list_projects(self):
        """Return [(id, title, description, ordering, created_at)] sorted
        by (ordering, id)."""
        return self.conn.execute(
            "SELECT id, title, description, ordering, created_at "
            "FROM projects ORDER BY ordering, id"
        ).fetchall()

    def get_project(self, project_id: int):
        return self.conn.execute(
            "SELECT id, title, description, ordering, created_at "
            "FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()

    def add_project(self, title: str, description: str = "",
                    ordering: int = 0) -> int:
        cur = self.conn.execute(
            "INSERT INTO projects (title, description, ordering) VALUES (?, ?, ?)",
            (title, description, ordering),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def update_project(self, project_id: int, *, title: str | None = None,
                       description: str | None = None,
                       ordering: int | None = None) -> bool:
        sets, vals = [], []
        if title is not None:
            sets.append("title = ?")
            vals.append(title)
        if description is not None:
            sets.append("description = ?")
            vals.append(description)
        if ordering is not None:
            sets.append("ordering = ?")
            vals.append(ordering)
        if not sets:
            return False
        vals.append(project_id)
        cur = self.conn.execute(
            f"UPDATE projects SET {', '.join(sets)} WHERE id = ?",
            vals,
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete_project(self, project_id: int) -> bool:
        cur = self.conn.execute(
            "DELETE FROM projects WHERE id = ?", (project_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    # ---- interaction counter --------------------------------------------

    def record_interaction(self, emp_id: str, session_id: str) -> bool:
        """Returns True iff this is a new (emp_id, session_id) pair."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO interactions (emp_id, session_id) VALUES (?, ?)",
            (emp_id, session_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def interaction_count_total(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM interactions"
        ).fetchone()
        return int(row[0]) if row else 0

    def interaction_count_session(self, session_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row[0]) if row else 0


def _to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _from_blob(buf: bytes) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.float32)
