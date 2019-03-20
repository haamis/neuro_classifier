from itertools import islice
import sys
import ijson

with open(sys.argv[1], "rt", encoding="utf-8") as f:
    #print(f.readlines())
    parser = ijson.parse(f)
    print(next(ijson.items(f, 'item.abstract')))
    print(next(ijson.items(f, 'item.abstract')))
    #for prefix, event, value in parser:
    #    print(prefix, event, value)