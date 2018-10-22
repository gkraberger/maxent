# See ../triqs/packaging for other options
FROM flatironinstitute/triqs:master-ubuntu-clang

RUN apt-get install -y python-decorator || yum install -y python-decorator
RUN apt-get install -y python-autopep8 || yum install -y python-autopep8

ARG APPNAME
COPY . $SRC/$APPNAME
WORKDIR $BUILD/$APPNAME
RUN chown build .
USER build
ARG BUILD_DOC=0
ARG USE_TRIQS=1
RUN PYTHONPATH=${USE_TRIQS:+$PYTHONPATH} CMAKE_PREFIX_PATH=${USE_TRIQS:+$CMAKE_PREFIX_PATH} cmake $SRC/$APPNAME -DUSE_TRIQS=${USE_TRIQS} -DTRIQS_ROOT=${USE_TRIQS:+$INSTALL} -DCMAKE_INSTALL_PREFIX=${INSTALL} -DBuild_Documentation=${BUILD_DOC} && make -j2 && make test
USER root
RUN make install
