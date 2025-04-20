FROM ollama/ollama

WORKDIR /root

COPY requirements.txt ./

# First install essential tools including wget
RUN apt-get update 
RUN apt-get install -y python3 python3-pip vim git wget build-essential

# Now we can download and install SQLite
RUN wget https://www.sqlite.org/2023/sqlite-autoconf-3420000.tar.gz \
    && tar -xvzf sqlite-autoconf-3420000.tar.gz \
    && cd sqlite-autoconf-3420000 \
    && ./configure --prefix=/usr \
    && make \
    && make install

# Install Python requirements
RUN pip install -r requirements.txt

EXPOSE 8501
EXPOSE 11434
ENTRYPOINT ["./entrypoint.sh"]

