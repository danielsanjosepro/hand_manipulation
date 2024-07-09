docker run \
    --net=host \
    --ipc=host \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --env DISPLAY \
    --privileged \
    -it hand_manipulation
