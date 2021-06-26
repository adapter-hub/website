#!/bin/bash

CACHE_FILE=".last_modified.cache"

python app/hf_hub.py $CACHE_FILE

# using exit code of hf_hub script
if [ $? -eq 1 ]
then
    export AH_SCHEDULED_CHANGE=0
else
    flask db init
    flask freeze build
    export AH_SCHEDULED_CHANGE=1
fi
