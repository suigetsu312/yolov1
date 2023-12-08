FROM tensorflow/tensorflow
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install opencv-python
