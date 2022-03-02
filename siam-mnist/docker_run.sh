#!/bin/bash 

! echo "Mounting docker in:" 
! pwd

sudo docker run --rm -it \
    --net host \
    --gpus=all \
    --user="$(id -u):$(id -g)" \
    --volume="$PWD:/app" \
    -e DISPLAY=${DISPLAY} \
    --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --device=/dev/video0:/dev/video0 \
  image-segmentation python3.8 src/main.py 