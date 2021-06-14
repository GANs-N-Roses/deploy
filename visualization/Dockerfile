FROM python:3.7
COPY . /visualization
WORKDIR /visualization
RUN apt-get update && apt-get -y install fluidsynth
RUN mkdir /usr/share/soundfonts && mv soundfonts/default.sf2 /usr/share/soundfonts/
RUN pip install -r requirements.txt && pip install -r pipelines/requirements.txt
CMD streamlit run /visualization/visualize.py --server.port $PORT