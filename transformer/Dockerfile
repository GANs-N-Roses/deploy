FROM python:3.7.10
RUN apt-get update -qq && apt-get install -qq libfluidsynth1 build-essential libasound2-dev libjack-dev
COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
# COPY ./requirements.txt /requirements.txt
# RUN pip install -r /requirements.txt
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin
COPY ./entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]