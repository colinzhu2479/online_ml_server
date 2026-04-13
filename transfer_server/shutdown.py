import Pyro4
import numpy as np
from sys import argv
import time
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
conn_f = config["server"]["connection_file"]
pw = config["server"]["password"]
if len(argv) == 1:
    connection = conn_f#'/N/project/sico/nn_server/connection.txt'
    with open(connection,'r') as f:
        uri = f.readline().strip()
        print(uri)
else:    
    uri = argv[1]

uri_s = Pyro4.Proxy(uri)    # use name server object lookup uri shortcut
uri_s._pyroHmacKey = pw
print(f'Shutdown request sent at {time.ctime()}.')
uri_s.shutdown()
