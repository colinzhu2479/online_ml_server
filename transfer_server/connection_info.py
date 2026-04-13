import os
import sys
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
os.chdir(script_dir)

try:
        import tomllib as toml
except:
        import toml
with open("server.toml","rb") as f:
        config = toml.load(f)

os.system(f'cat {config["server"]["connection_file"]}')
