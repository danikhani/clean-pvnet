# README

## Build 

```bash
docker build -t pvnet_ndds:latest .
```
## Run

To run the docker
Add the following to your ~/.bashrc

```bash
export PVNET_DOCKER=pvnet_ndds:latest
export PVNET_GIT=$HOME/path/to/clean-pvnet # update this path to your downloaded repo
source $PVNET_GIT/docker/setup_dev.bash
```

run it with:

```bash
pvnet_ndds
```
