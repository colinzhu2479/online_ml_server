# Online ML server for molecular dynamics speed up

Currently developing package.

An independent client-server style prediction architecture in which an online ML server provides asynchronous force predictions and model versioning. The server operates independently of the molecular dynamics engine and supports multiple concurrent simulations, facilitating scalable deployment, centralized model management, and efficient reuse of learned representations.

![Architecture Diagram](docs/images/server.png)


---

## Launch on HPC
1. Edit server connection and communication settings in
```
transfer_server/server.toml
```
2. Edit resource configuration for server submitted as a batch job
```
transfer_server/launch_server.script
```
3. Submit job in transfer_server
```
sbatch launch_server.script
```
or directly run nn_host.py in interactive section
```
python nn_host.py
```

## Dependencies
See requirement.txt

## Module Reference
Remote server instruction:
```
python transfer_server/check_status.py ### check server running status
python transfer_server/clear_status.py ### reset prediction counts
python transfer_server/connection_info.py ### check IP address, connection, server launch time
python transfer_server/shutdown.py ### shutdown server
```

Toolkit:
```
toolkit/save_cluster.py
toolkit/combine_training.py
```
