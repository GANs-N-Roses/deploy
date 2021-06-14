Pipelines for our AI
====================

## image_pipe
For image model execution. Given a path to an image, passes it through the net to get an emotion.

## intermediate_pipe
To convert the previous predictions into a something like valence-arousal to get the most similar song considering the valence-arousal vector.

## music_pipe
Given an initial song, generate a new song.

## every_pipe
Integrates each and every pipe in just a single pipe