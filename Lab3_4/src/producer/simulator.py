import json, random, time, yaml
from itertools import count
from datetime import datetime, timezone
from confluent_kafka import Producer
from src.utils.logger import get_logger

log = get_logger("SimProducer")

def load_yaml(p):
    with open(p,"r",encoding="utf-8") as f: return yaml.safe_load(f)

def load_data(path):
    with open(path,"r",encoding="utf-8") as f:
        data=[json.loads(l) for l in f if l.strip()]
    return data

EMOJIS = ["🙂","😊","🔥","💯","😡","😞","👍","👎","✨","🚀","🐞","☀️","❄️"]
HASHTAGS = ["#update","#service","#app","#coffee","#weather","#release","#bugfix","#news","#ai"]
MENTIONS = ["@support","@devteam","@product","@qa"]

def augment(base):
    text = base["text"]
    if random.random()<0.3: text += " " + random.choice(HASHTAGS)
    if random.random()<0.3: text += " " + random.choice(EMOJIS)
    if random.random()<0.15: text += " " + random.choice(MENTIONS)
    if random.random()<0.15: text += " http://example.com/x"
    return text

def run(kafka_cfg="configs/kafka.yaml", app_cfg="configs/app.yaml", data_path="data/tweets.jsonl"):
    kc = load_yaml(kafka_cfg); ac = load_yaml(app_cfg)
    p = Producer({"bootstrap.servers": kc["bootstrap_servers"]})
    topic = kc["topics"]["raw"]

    data = load_data(data_path)
    next_id = count(start=max([d.get("id",0) for d in data]+[0])+1)

    def delay():
        base = ac["simulate"]["interval_ms"]/1000.0
        jitter = ac["simulate"].get("jitter_ms",0)/1000.0
        if jitter>0:
            return max(0.0, base + random.uniform(-jitter, jitter))
        return base

    def one_pass():
        items = data[:]
        if ac["simulate"].get("shuffle", True):
            random.shuffle(items)
        for item in items:
            o = dict(item)
            o["id"] = next(next_id)
            o["ts"] = datetime.now(timezone.utc).isoformat()
            o["text"] = augment(o)
            p.produce(topic, json.dumps(o, ensure_ascii=False).encode("utf-8"))
            log.info(f"Produced: {o}")
            p.poll(0)
            time.sleep(delay())
        p.flush()

    if ac["simulate"].get("continuous", True):
        while True:
            one_pass()
    else:
        one_pass()

if __name__=="__main__": run()