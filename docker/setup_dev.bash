#!/bin/bash

pvnet_docker() {
  echo "starting pvnet_NDDS docker"
  xhost +local:docker;
  docker run -it --rm -d \
    --runtime=nvidia \
    --name="pvnetn_dev" \
    -v /etc/localtime:/etc/localtime:ro \
    -v /dev/input:/dev/input \
    -v "$HOME/docker_shared:$HOME/docker_shared" \
    -v "$PVNET_GIT:$HOME/pvnet" \
    --shm-size=4G \
    --workdir $HOME/ \
    --net=host \
    --add-host pvnetn_dev:127.0.0.1 \
    --hostname=pvnetn_dev \
    --privileged=true \
    --env=DISPLAY \
    --env=XDG_RUNTIME_DIR \
    --env=QT_X11_NO_MITSHM=1 \
    --device=/dev/dri:/dev/dri \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /etc/localtime:/etc/localtime:ro \
    $PVNET_DOCKER

    pvnet_docker_attach;
}

pvnet_docker_attach() {
  docker exec -it -e "COLUMNS=$COLUMNS" -e "LINES=$LINES" pvnetn_dev /bin/bash
}
