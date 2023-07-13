FROM ubuntu:22.04
# Ensure system is up-to-date.
RUN apt-get -y update 
RUN apt-get -y upgrade
# https://idroot.us/install-mesa-drivers-ubuntu-22-04/
# https://stackoverflow.com/questions/32486779/apt-add-repository-command-not-found-error-in-dockerfile
RUN apt-get -y install wget apt-transport-https gnupg2 software-properties-common
RUN apt-get -y install mesa-utils
# Add mesa repo.
# https://www.linuxcapable.com/how-to-upgrade-mesa-drivers-on-ubuntu-linux/
RUN add-apt-repository ppa:kisak/kisak-mesa -y
# Ensure system is up-to-date again (updating mesa).
RUN apt-get -y update && apt-get -y upgrade
# Add the vulkan stuff.
RUN apt-get -y install libvulkan1 vulkan-tools mesa-vulkan-drivers 
# install build tools
RUN apt-get -y install python3 cmake nodejs ninja-build build-essential wget curl
# install other essentials
RUN apt-get -y install less parallel vim
# install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"
# Setup the volume
RUN mkdir /work
VOLUME /work
WORKDIR /work
# Drop into bash.
CMD ["/bin/bash"]
