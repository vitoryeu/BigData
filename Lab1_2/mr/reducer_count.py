import sys
cur=None; acc=0
for raw in sys.stdin:
    parts=raw.rstrip("\n").split("\t")
    if len(parts)!=2: continue
    k,v=parts[0],int(parts[1])
    if cur==k: acc+=v
    else:
        if cur is not None: print("{0}\t{1}".format(cur,acc))
        cur=k; acc=v
if cur is not None: print("{0}\t{1}".format(cur,acc))
