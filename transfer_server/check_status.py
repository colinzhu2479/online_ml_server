import Pyro4
import numpy as np
from sys import argv
import sys
import os
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
os.chdir(script_dir)

try:
    import tomllib as toml
except:
    import toml
with open("server.toml","rb") as f:
    config = toml.load(f)

if len(argv) == 1:
    connection = config["server"]["connection_file"]#'/N/project/sico/nn_server/connection.txt'
    with open(connection,'r') as f:
        uri = f.readline().strip()
        print(uri)
else:    
    uri = argv[1]

uri_s = Pyro4.Proxy(uri)    # use name server object lookup uri shortcut
uri_s._pyroHmacKey = config["server"]["password"]
print(uri_s.get_status())
