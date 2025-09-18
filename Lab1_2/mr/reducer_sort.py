import sys
rows=[]
for raw in sys.stdin:
    parts=raw.rstrip("\n").split("\t")
    if len(parts)!=2: continue
    rows.append((parts[0], int(parts[1])))
for p,c in sorted(rows, key=lambda x: x[1], reverse=True):
    print("{0}\t{1}".format(p,c))
