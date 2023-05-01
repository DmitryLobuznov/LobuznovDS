#!/bin/bash

wget http://nlp.stanford.edu/data/glove.6B.zip
unzip -l glove.6B.zip
unzip glove.6B.zip glove.6B.50d.txt -d ./data
rm glove.6B.zip
echo "Embeddings loaded successfully!"
