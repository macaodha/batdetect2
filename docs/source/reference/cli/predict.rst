Predict command
===============

Use ``batdetect2 predict`` to run prediction on audio.

Choose a subcommand based on how you want to provide the input:

- ``directory`` for all supported audio files in one folder
- ``file_list`` for a text file with one audio path per line
- ``dataset`` for recordings referenced by a dataset file

Use ``--detection-threshold`` when you want to override the configured
threshold for one run.

.. click:: batdetect2.cli.inference:predict
   :prog: batdetect2 predict
   :nested: full
