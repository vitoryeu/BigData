from flask import Flask, render_template_string
from confluent_kafka import Consumer
import json, threading, queue

BOOTSTRAP="127.0.0.1:9092"
TOPIC="sentiments"
GROUP="bd-dashboard"

app = Flask(__name__)
q = queue.Queue(); rows=[]; stats={"total":0,"positive":0,"neutral":0,"negative":0}

HTML='''<!doctype html><html><head><meta charset="utf-8"><title>Kafka Dashboard</title>
<meta http-equiv="refresh" content="2">
<style>body{font-family:system-ui;margin:20px} table{border-collapse:collapse;width:100%} td,th{border:1px solid #eee;padding:6px}</style>
</head><body>
<h3>Kafka Sentiment Dashboard</h3>
<p>Total: {{s.total}} | Positive: {{s.positive}} | Neutral: {{s.neutral}} | Negative: {{s.negative}}</p>
<table><thead><tr><th>ID</th><th>Lang</th><th>Sentiment</th><th>Score</th><th>Text</th></tr></thead>
<tbody>{% for r in rows[-100:] %}<tr><td>{{r["id"]}}</td><td>{{r["lang"]}}</td><td>{{r["sentiment"]}}</td><td>{{r["score"]}}</td><td>{{r["text"]}}</td></tr>{% endfor %}</tbody>
</table></body></html>'''

def consume():
    c = Consumer({"bootstrap.servers": BOOTSTRAP, "group.id": GROUP, "auto.offset.reset": "earliest"})
    c.subscribe([TOPIC])
    try:
        while True:
            m = c.poll(1.0)
            if m is None: continue
            if m.error(): continue
            q.put(json.loads(m.value().decode("utf-8")))
    finally:
        c.close()

@app.route("/")
def index():
    while not q.empty():
        v=q.get(); rows.append(v); stats["total"]+=1; s=v.get("sentiment","neutral")
        if s in stats: stats[s]+=1
    return render_template_string(HTML, s=stats, rows=rows)

def main():
    threading.Thread(target=consume, daemon=True).start()
    app.run("127.0.0.1", 5000, debug=True)

if __name__=="__main__": main()
