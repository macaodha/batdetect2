Finetune command
================

Use ``batdetect2 finetune`` to adapt an existing checkpoint to a new target
definition.

If you do not pass ``--model``, the bundled ``uk_same`` checkpoint is used.

.. click:: batdetect2.cli.finetune:finetune_command
   :prog: batdetect2 finetune
   :nested: none
