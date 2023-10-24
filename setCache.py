import sys

if len(sys.argv) < 2:
    print("Usage: python setCache.py [value]")
    exit()

value = int(sys.argv[1])

with open("utils/constants.py", "w") as f:
    f.write("CACHE_SIZE = {}\n".format(value))
