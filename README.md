# FLCSDet
We have developed a federated learning-driven maritime visual surveillance method for multi-agency coordinated surveillance of marine areas.

Server

In order to start a Federated Learning training, it is necessary to start the server:

python fd_server.py --config_file yolo_task.json --port 12345

port: ip of the server
config_file: parameters necessary for federated learning
Client

Clients must be created after the server.
python fd_client1.py
python fd_client2.py
python fd_client3.py
