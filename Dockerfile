FROM phusion/baseimage:0.11
RUN apt-get update -y
RUN apt-get install python3 -y
RUN apt-get install python3-pip -y
RUN apt-get install python-gevent -y
RUN apt-get install python-gevent-websocket -y
RUN apt-get install git -y
RUN apt-get install libgtk2.0-dev  -y
RUN apt-get install libsm6 -y
RUN apt-get install libxext6 -y
RUN apt-get install iputils-ping -y
RUN pip3 install shapely
RUN pip3 install imutils
RUN pip3 install gevent
RUN pip3 install opencv-python

RUN mkdir /etc/service/nokia_cv
RUN echo "#!/bin/sh" > /etc/service/nokia_cv/run
RUN echo "python3 /nokia_iot_cvision/detector.py --source \`cat /etc/container_environment/VIDEO_SOURCE\`" >> /etc/service/nokia_cv/run
RUN chmod +x /etc/service/nokia_cv/run

ARG CACHEBUST=1
RUN git clone https://github.com/glial-iot/nokia_iot_cvision.git
RUN mkdir nokia_iot_cvision/data

RUN apt-get upgrade -y -o Dpkg::Options::="--force-confold"
RUN apt-get remove git python3-pip -y
RUN apt-get autoremove -y
RUN apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV KILL_PROCESS_TIMEOUT=300
ENV KILL_ALL_PROCESSES_TIMEOUT=300
VOLUME [ "/nokia_iot_cvision/data" ]
CMD ["/sbin/my_init"]
