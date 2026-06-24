import sys

from dev.public_export.sync import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "sync":
        argv = argv[1:]
    raise SystemExit(main(argv))
