Evaluate command
================

Use ``batdetect2 evaluate`` to compare a checkpoint against labelled test data.

This command writes metrics and any configured artifacts to the output
directory.

.. click:: batdetect2.cli.evaluate:evaluate_command
   :prog: batdetect2 evaluate
   :nested: none
