# README

## Build 

```bash
docker build -t pvnet_ndds:latest .
```
system : Successfully tagged pvnet_ndds:latest
## Run

To run the docker
Add the following to your ~/.bashrc

```bash
export PVNETNDDS_DOCKER=pvnet_ndds:latest
export PVNETNDDS_GIT=$HOME/Documents/RWTH/ATP/linemod_understanding/katharina/try4_docker/clean-pvnet  # update
source $PVNETNDDS_GIT/docker/setup_dev.bash
```

run it with:

```bash
pvnet_ndds
```
