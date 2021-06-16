#!/bin/bash

CACHE_FILE=".last_modified.cache"

python app/hf_hub.py $CACHE_FILE

# using exit code of hf_hub script
if [ $? -eq 1 ]
then
    exit 1
else
    flask db init
    flask freeze build
    exit 0
fi
