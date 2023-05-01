#!/bin/bash

wget https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip
unzip -l QQP-clean.zip
unzip QQP-clean.zip -d ./data
rm QQP-clean.zip
echo "QQP data loaded successfully!"