FROM quay.io/jupyter/scipy-notebook:2024-04-01

USER root

# install gdal
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#    gdal-bin libgdal-dev proj-bin libproj-dev \
#    && \
#    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libeccodes-dev \
    && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

# Install from the requirements.txt file
COPY --chown=${NB_UID}:${NB_GID} requirements.txt /tmp/
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"