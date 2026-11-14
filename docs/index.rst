Welcome to Triton's documentation!
==================================

Triton_ is a language and compiler for parallel programming. It aims to provide a Python-based programming environment for productively writing custom DNN compute kernels capable of running at maximal throughput on modern GPU hardware.


Getting Started
---------------

- Follow the :doc:`installation instructions <getting-started/installation>` for your platform of choice.
- Take a look at the :doc:`tutorials <getting-started/tutorials/index>` to learn how to write your first Triton program.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting-started/installation
   getting-started/tutorials/index


Python API
----------

- :doc:`triton <python-api/triton>`
- :doc:`triton.language <python-api/triton.language>`
- :doc:`triton.testing <python-api/triton.testing>`
- :doc:`Operator graph fusion with Inductor <python-api/operator-fusion>`
- :doc:`Intermediate global-memory elimination <python-api/intermediate-memory-elimination>`
- :doc:`Experimental model graph fusion <python-api/graph-fusion>`
- :doc:`Experimental XPU resource partitions <python-api/graph-fusion-resources>`
- :doc:`P800 resource and partition autotuning <python-api/resource-autotuning>`
- :doc:`Triton semantics <python-api/triton-semantics>`
- :doc:`triton.language.extra.cuda <python-api/triton.language.extra.cuda>`


.. toctree::
   :maxdepth: 1
   :caption: Python API
   :hidden:

   python-api/triton
   python-api/triton.language
   python-api/triton.testing
   python-api/operator-fusion
   python-api/intermediate-memory-elimination
   python-api/graph-fusion
   python-api/graph-fusion-resources
   python-api/resource-autotuning
   python-api/triton-semantics


Triton MLIR Dialects and Ops
--------------------

- :doc:`Triton MLIR Dialects and Ops <dialects/dialects>`

.. toctree::
   :maxdepth: 1
   :caption: Triton MLIR Dialects
   :hidden:

   dialects/dialects

Going Further
-------------

Check out the following documents to learn more about Triton and how it compares against other DSLs for DNNs:

- Chapter 1: :doc:`Introduction <programming-guide/chapter-1/introduction>`
- Chapter 2: :doc:`Related Work <programming-guide/chapter-2/related-work>`
- Chapter 3: :doc:`Debugging <programming-guide/chapter-3/debugging>`

.. toctree::
   :maxdepth: 1
   :caption: Programming Guide
   :hidden:

   programming-guide/chapter-1/introduction
   programming-guide/chapter-2/related-work
   programming-guide/chapter-3/debugging

.. _Triton: https://github.com/triton-lang/triton
