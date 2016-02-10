FROM yandex/rep:0.6.5

RUN /bin/bash --login -c " \
  pip install seaborn \
  "
