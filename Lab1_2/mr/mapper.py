import sys, re
pattern = re.compile(r'(\S+) - - \[.*?\] "\S+ (\S+)')
for line in sys.stdin:
    m = pattern.match(line)
    if m:
        print("{0}\t1".format(m.group(2)))
