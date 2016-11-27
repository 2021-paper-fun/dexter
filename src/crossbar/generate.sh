#!/bin/bash

openssl genrsa -out .crossbar/server_key.pem 2048
chmod 600 .crossbar/server_key.pem
openssl req -new -key .crossbar/server_key.pem -out .crossbar/server_csr.pem
openssl x509 -req -days 365 -in .crossbar/server_csr.pem \
    -signkey .crossbar/server_key.pem -out .crossbar/server_cert.pem
