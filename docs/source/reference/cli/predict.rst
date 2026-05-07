Process command
===============

Use ``batdetect2 process`` to run inference on audio.

Choose a subcommand based on how you want to provide the input:

- ``directory`` for all supported audio files in one folder
- ``file_list`` for a text file with one audio path per line
- ``dataset`` for recordings referenced by a dataset file

Use ``--detection-threshold`` when you want to override the configured
threshold for one run.

.. click:: batdetect2.cli.inference:process
   :prog: batdetect2 process
   :nested: full
