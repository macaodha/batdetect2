Train command
=============

Use ``batdetect2 train`` to start from a fresh model config or continue from an
existing checkpoint.

If you want to adapt an existing checkpoint to a new target definition, use
``batdetect2 finetune`` instead.

.. click:: batdetect2.cli.train:train_command
   :prog: batdetect2 train
   :nested: none
