#!/usr/bin/env bash

if [ "$#" -eq 0 ] || [ "$1" == outer ]; then
    # Run the podman container.
    podman run -it -e DISPLAY --net=host --device=/dev/dri:/dev/dri --mount type=bind,source=.,target=/work griddly
elif [ "$1" == inner ]; then
    # Go into the project directory
    cd /work 
    # Install the virtual environment.
    poetry install
    # Fix passive_env_checker to get rid of a deprecation warning.
    sed -i 's/bool8/bool_/g' /root/.cache/pypoetry/virtualenvs/hsp-DJpFP61h-py3.10/lib/python3.10/site-packages/gym/utils/passive_env_checker.py
fi
