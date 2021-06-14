# Music Transformer

This is the environment for GANs & Roses music generation model, based on [Magenta Music Transformer](https://magenta.tensorflow.org/music-transformer)

## Prerequisites

* Docker [INSTALL](https://www.docker.com/get-started)

## Set up

Clone repository:

```bash
git clone https://github.com/GANs-N-Roses/transformer.git
```

Move to the project directory:

```bash
cd transformer
```

Build the image (this should take around 3.6 GB in disk):

```bash
docker-compose build
```

Run it.

```bash
docker-compose run
```

It will download the music transformer pretrained models (only the first time on the first run) and then start the container.

When it's done, it will print the URL to access the Jupyter Notebook from the host machine:

```
roses  |      or http://127.0.0.1:8888/?token=c42836b292095cb6e374e09e6765e90ae3455338ba7ca842
```

Also, it's possible to access the container bash directly:

Check the container name using
```
docker ps
```

And then:
```
docker exec -it your-container-name bash
```

## Usage

Your local ```src``` directory is synced with the Jupyter Notebook's working directory (/music), so any files in this directory will be visible within the Docker container and ready to be loaded to Jupyter. Similarly, any Notebook you save will appear in this directory.

Learn how to use the music transformer with the ```demo.ipynb``` Notebook (in ```src```).

Also, you may add or remove dependancies by modifying ```requirements.txt```.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
