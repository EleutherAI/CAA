#!/bin/bash

# make sure $1 isn't empty
if [ -z "$1" ]; then
    echo "Please provide a name for the archive folder"
    exit 1
fi

cp -a vectors erasers normalized_vectors results analysis output/$1

# if $2 is -c, also run ./cleanup.sh
if [ "$2" = "-c" ]; then
    ./cleanup.sh
fi