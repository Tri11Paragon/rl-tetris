import sys

import config

def main(config_path):
    config.Config(config_path).load()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python typegen.py <config_path>")
        sys.exit(1)
    main(sys.argv[1])