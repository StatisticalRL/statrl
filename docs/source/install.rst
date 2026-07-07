Installation
============

``statrl`` requires **Python 3.9+**. It depends on ``numpy``, ``gymnasium``, ``joblib``,
and ``matplotlib`` (installed automatically).

From source
-----------

The package is not yet on PyPI. Clone the repository and install it:

.. code-block:: bash

   git clone https://github.com/StatisticalRL/statrl.git
   cd statrl
   pip install .

For development, install in editable mode so code changes take effect immediately:

.. code-block:: bash

   pip install -e .

Verify the installation
-----------------------

.. code-block:: bash

   python -c "import statrl.settings.bandits.stochastic.anytime.agents.IMED as m; print(m.IMED)"

Building the documentation
--------------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs/source docs/_build/html
   # open docs/_build/html/index.html
