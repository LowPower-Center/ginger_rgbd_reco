FROM ultralytics/yolov5:latest-cpu
RUN pip install --no-cache flask 
COPY best.pt /usr/src/app
COPY taiqiu-vision.sh /usr/src/app
COPY restapi.py /usr/src/app
WORKDIR /usr/src/app
EXPOSE 4003
RUN chmod +x taiqiu-vision.sh
#CMD [ "ls" ]
CMD [ "./taiqiu-vision.sh" ]