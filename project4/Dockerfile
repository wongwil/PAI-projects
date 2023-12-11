FROM mambaorg/micromamba:latest
RUN micromamba install -y -n base python=3.8.5 -c conda-forge && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)
WORKDIR /code
ADD * /code/
ADD pytransform /code/pytransform
ADD requirements.txt /code/requirements.txt
RUN pip install -r requirements.txt
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/code/pytransform
WORKDIR /code
CMD python -u checker_client.py
