FROM ollama/ollama

WORKDIR /root

COPY requirements.txt ./

RUN apt update 
RUN apt-get install -y python3 python3-pip vim git
RUN pip install -r requirements.txt

# Install SQLite3 from source
RUN wget https://www.sqlite.org/2023/sqlite-autoconf-3420000.tar.gz \
    && tar -xvzf sqlite-autoconf-3420000.tar.gz \
    && cd sqlite-autoconf-3420000 \
    && ./configure --prefix=/usr \
    && make \
    && make install

EXPOSE 8501
EXPOSE 11434
ENTRYPOINT ["./entrypoint.sh"]

