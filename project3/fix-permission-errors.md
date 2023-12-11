How to fix permission errors
============================

In some specific setups, running `bash runner.sh` might fail to write a `results_check.byte` file.
If running your task displays a score, but then fails with a message such as

    Error: Could not write results file

or

    PermissionError: [Errno 13] Permission denied: '/results/results_check.byte'

you can try one of the following two workarounds.
If neither workaround resolves your issues then please check Moodle,
and post a new question if there is no solution.

**Both workarounds assume that you are using Docker on Linux.**


Workaround 1: fixing directory permissions
------------------------------------------

The following is an easy fix that resolves permission errors,
but can break if you copy/move your handout directory: 

1. Open a shell and navigate to the handout directory.
   Running `ls` should list `Dockerfile`, `solution.py`, `runner.sh`, and all other handout files.
2. In that shell, run `chmod 777 .`

After that, all users should be able to create, write, and delete files in the handout directory.


Workaround 2: fixing Docker container user
------------------------------------------

This workaround is more complicated,
but does not rely on directory permissions.
Hence, the fix should persist
if you move/copy the handout directory.

1. Make sure to use a freshly downloaded handout
2. If you already started implementing a solution, copy-paste _only_ your `solution.py` into the newly downloaded handout directory
3. In `Dockerfile`, directly after `FROM ...`, place the following code snippet:
```
ARG NEW_MAMBA_USER
ARG NEW_MAMBA_USER_ID
ARG NEW_MAMBA_USER_GID
USER root

RUN if grep -q '^ID=alpine$' /etc/os-release; then \
      # alpine does not have usermod/groupmod
      apk add --no-cache --virtual temp-packages shadow; \
    fi && \
    usermod "--login=${NEW_MAMBA_USER}" "--home=/home/${NEW_MAMBA_USER}" \
        --move-home "-u ${NEW_MAMBA_USER_ID}" "${MAMBA_USER}" && \
    groupmod "--new-name=${NEW_MAMBA_USER}" \
        "-g ${NEW_MAMBA_USER_GID}" "${MAMBA_USER}" && \
    if grep -q '^ID=alpine$' /etc/os-release; then \
      # remove the packages that were only needed for usermod/groupmod
      apk del temp-packages; \
    fi && \
    # Update the expected value of MAMBA_USER for the
    # _entrypoint.sh consistency check.
    echo "${NEW_MAMBA_USER}" > "/etc/arg_mamba_user" && \
    :
ENV MAMBA_USER=$NEW_MAMBA_USER
USER $MAMBA_USER
# Rest of the Dockerfile
```
4. Change the contents of `runner.sh` to the following, making sure to replace the two instances of `taskX`:
```
docker build --tag taskX --build-arg NEW_MAMBA_USER="$( id -un )" --build-arg NEW_MAMBA_USER_ID="$( id -u )" --build-arg NEW_MAMBA_USER_GID="$( id -g )" . && \
  docker run --rm -v "$( cd "$( dirname "$0" )" && pwd )":/results taskX
```
