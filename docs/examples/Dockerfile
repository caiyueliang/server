# FROM nvcr.io/nvidia/tritonserver:23.02-py3
FROM nvcr.io/nvidia/tritonserver:21.12-py3

# USER root

# COPY ./app/requirements.txt /tmp/requirements.txt

# RUN pip install -r /tmp/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
# RUN pip install requests==2.22.0 Flask-Limiter==1.5 -i https://pypi.tuna.tsinghua.edu.cn/simple/

#COPY ./app /home/server/app
#COPY ./babel /home/server/babel
COPY ./model_repository /models

#WORKDIR /home/server

## CMD ["python", "run.py"]
CMD ["tritonserver", "--model-repository=/models"]
