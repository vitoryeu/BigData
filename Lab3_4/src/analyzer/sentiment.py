# sentiment using confluent-kafka
import json, yaml
from confluent_kafka import Consumer, Producer
from src.utils.logger import get_logger

log = get_logger("Analyzer")

POS = {"love","great","good","amazing","friendly","чудова","супер"}
NEG = {"terrible","awful","bad","slow","never again","багів","поган"}

def score_text(t: str):
    l=t.lower(); p=sum(1 for w in POS if w in l); n=sum(1 for w in NEG if w in l)
    if p==n==0: return "neutral",0.0
    if p>n: return "positive",(p-n)/(p+n)
    if n>p: return "negative",(n-p)/(p+n)
    return "neutral",0.0

def load_yaml(p):
    with open(p,"r",encoding="utf-8") as f: return yaml.safe_load(f)

def run(kafka_cfg="configs/kafka.yaml"):
    kc = load_yaml(kafka_cfg)
    c = Consumer({
        "bootstrap.servers": kc["bootstrap_servers"],
        "group.id": kc["consumer"]["group_id"],
        "auto.offset.reset": kc["consumer"]["auto_offset_reset"]
    })
    c.subscribe([kc["topics"]["clean"]])
    p = Producer({"bootstrap.servers": kc["bootstrap_servers"]})
    out_topic = kc["topics"]["sentiments"]
    try:
        while True:
            msg = c.poll(1.0)
            if msg is None: continue
            if msg.error():
                log.warning(f"Consumer error: {msg.error()}")
                continue
            item = json.loads(msg.value().decode("utf-8"))
            s, sc = score_text(item["text"])
            out = {"id": item.get("id"), "lang": item.get("lang","en"), "text": item["text"], "sentiment": s, "score": round(float(sc),4)}
            p.produce(out_topic, json.dumps(out, ensure_ascii=False).encode("utf-8"))
            log.info(f"Analyzed: {out}")
            p.poll(0)
    finally:
        c.close(); p.flush()

if __name__=="__main__": run()
