docker build --tag task3 . && \
  docker run --rm -u $(id -u):$(id -g) -v "$( cd "$( dirname "$0" )" && pwd )":/results task3
