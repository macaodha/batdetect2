Legacy detect command
=====================

.. warning::

   ``batdetect2 detect`` is a legacy compatibility command.
   Prefer ``batdetect2 predict directory`` for new workflows.

Migration at a glance
---------------------

- Legacy: ``batdetect2 detect AUDIO_DIR ANN_DIR DETECTION_THRESHOLD``
- Current: ``batdetect2 predict directory MODEL_PATH AUDIO_DIR OUTPUT_PATH``
  with optional ``--detection-threshold``

.. click:: batdetect2.cli.compat:detect
   :prog: batdetect2 detect
   :nested: none
