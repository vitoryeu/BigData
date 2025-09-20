# cleaner using confluent-kafka
import json, yaml
from confluent_kafka import Consumer, Producer
from src.utils.logger import get_logger
from src.utils.text_utils import clean_text

log = get_logger("Processor")

def load_yaml(p):
    with open(p,"r",encoding="utf-8") as f: return yaml.safe_load(f)

def run(kafka_cfg="configs/kafka.yaml", app_cfg="configs/app.yaml"):
    kc = load_yaml(kafka_cfg); ac = load_yaml(app_cfg)
    c = Consumer({
        "bootstrap.servers": kc["bootstrap_servers"],
        "group.id": kc["consumer"]["group_id"],
        "auto.offset.reset": kc["consumer"]["auto_offset_reset"]
    })
    c.subscribe([kc["topics"]["raw"]])
    p = Producer({"bootstrap.servers": kc["bootstrap_servers"]})
    out_topic = kc["topics"]["clean"]
    try:
        while True:
            msg = c.poll(1.0)
            if msg is None: continue
            if msg.error(): 
                log.warning(f"Consumer error: {msg.error()}"); 
                continue
            item = json.loads(msg.value().decode("utf-8"))
            if not isinstance(item, dict) or "text" not in item: 
                continue
            item["text"] = clean_text(item["text"], **ac["text"])
            p.produce(out_topic, json.dumps(item, ensure_ascii=False).encode("utf-8"))
            log.info(f"Cleaned: {item}")
            p.poll(0)
    finally:
        c.close(); p.flush()

if __name__=="__main__": run()
