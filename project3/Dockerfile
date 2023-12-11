FROM mambaorg/micromamba:latest
RUN micromamba install -y -n base python=3.8.5 -c conda-forge && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)
RUN pip install pyarmor==6.7.4 scipy==1.7 cryptography==39.* \
scikit-learn==1.3.* matplotlib==3.4.*
WORKDIR /code
ADD * /code/
ADD pytransform /code/pytransform
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/code/pytransform
WORKDIR /code
CMD python -u checker_client.py
