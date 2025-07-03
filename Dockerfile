FROM vastai/pytorch:2.5.1-cuda-12.1.1

WORKDIR /tmp
COPY install-miniconda.sh ./
RUN bash install-miniconda.sh
COPY . /workspace/code/Mask3D/
RUN echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda-11.3/targets/x86_64-linux/lib" >> /root/.bashrc
