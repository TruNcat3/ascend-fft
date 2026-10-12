#!/usr/bin/env python3
"""Extract kernel timing from msprof DB. Usage: python3 extract-kernel-timing.py <msprof.db>"""
import sqlite3, sys

def extract(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT DISTINCT t.taskType FROM TASK t JOIN COMPUTE_TASK_INFO ci ON t.globalTaskId=ci.globalTaskId LIMIT 1')
    row = c.fetchone()
    if not row:
        print("No COMPUTE_TASK_INFO found"); return
    tt = row[0]
    c.execute('''
        SELECT si.value, COUNT(*), AVG(t.endNs - t.startNs)/1000.0
        FROM TASK t
        JOIN COMPUTE_TASK_INFO ci ON t.globalTaskId = ci.globalTaskId
        JOIN STRING_IDS si ON ci.name = si.id
        WHERE t.taskType = ?
        GROUP BY si.value ORDER BY AVG(t.endNs - t.startNs) DESC
    ''', (tt,))
    for name, cnt, avg_us in c.fetchall():
        print(f"  {name}: {cnt} tasks, avg {avg_us:.1f}us")
    conn.close()

if __name__ == "__main__":
    extract(sys.argv[1])
