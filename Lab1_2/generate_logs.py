import random
from datetime import datetime, timedelta
import ipaddress, os

PAGES = ['/', '/home', '/about', '/products', '/contact', '/blog', '/services',
         '/login', '/register', '/cart', '/checkout', '/profile', '/settings']
METHODS = ['GET', 'POST']
STATUS = [200, 301, 302, 304, 400, 401, 403, 404, 500]

def one(start, end):
    ts = start + timedelta(seconds=random.randint(0, int((end-start).total_seconds())))
    ip = str(ipaddress.IPv4Address(random.randint(0, 2**32-1)))
    method = random.choice(METHODS)
    path = random.choice(PAGES)
    proto = 'HTTP/1.1'
    status = random.choice(STATUS)
    size = random.randint(200, 50000)
    ts_str = ts.strftime('%d/%b/%Y:%H:%M:%S +0000')
    return f"{ip} - - [{ts_str}] \"{method} {path} {proto}\" {status} {size}"

if __name__ == "__main__":
    os.makedirs("data/input", exist_ok=True)
    out = "data/input/weblog.log"
    start = datetime(2024,1,1); end = datetime(2024,12,31,23,59,59)
    with open(out, "w", encoding="utf-8") as f:
        for _ in range(200000):
            f.write(one(start, end) + "\n")
    print("Wrote to", out)
