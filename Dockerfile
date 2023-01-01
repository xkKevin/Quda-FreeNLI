FROM registry.zjvis.org/xiongkai/freenli:mlabel.1
# 如该目录不存在，WORKDIR 会帮你建立目录。
WORKDIR /freeNLI
COPY ./ /freeNLI
EXPOSE 80
# RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple django==3.0.2 pandas xlrd

# RUN ". activate simpletransformers"

# RUN rm /bin/sh && ln -s /bin/bash /bin/sh
# RUN "source activate simpletransformers"

# conda env list 执行的结果会显示在build构建的过程中
# RUN /bin/bash -c "source activate simpletransformers" && conda env list

# ~/.bashrc 可以在打开窗口（重启容器）时自动执行该脚本中的命令
# RUN echo "source activate simpletransformers" >> ~/.bashrc

RUN echo "source activate simpletransformers" >> /start.sh \
    && echo 'pip install -i https://pypi.tuna.tsinghua.edu.cn/simple lxml' >> /start.sh \
    && echo 'python ./manage.py runserver 0.0.0.0:80' >> /start.sh \
    && chmod 777 /start.sh

CMD ["bash", "/start.sh"]
# CMD ["python", "manage.py", "runserver", "0.0.0.0:80"]