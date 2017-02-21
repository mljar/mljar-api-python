|Build Status| |PyPI version| |Coverage Status|

mljar-api-python
================

A simple python wrapper over mljar API. It allows MLJAR users to create
Machine Learning models with few lines of code:

.. code:: python

    from mljar import Mljar

    model = Mljar(project='My awesome project', experiment='First experiment')
    model.fit(X,y)

    model.predict(X)

That's all folks! Yeah, I know, this makes Machine Learning super easy!
You can use this code for following Machine Learning tasks: \* Binary
classification (your target has only two unique values) \* Regression
(your target value is continuous) \* More is coming soon!

How to install
--------------

You can install mljar with **pip**:

::

    pip install -U mljar

or from source code:

::

    python setup.py install

How to use it
-------------

1. Create an account at mljar.com and login.
2. Please go to your users settings (top, right corner).
3. Get your token, for example 'exampleexampleexample'.
4. Set environment variable ``MLJAR_TOKEN`` with your token value:

   ::

       export MLJAR_TOKEN=exampleexampleexample

5. That's all, you are ready to use MLJAR in your python code!

What's going on?
----------------

-  This wrapper allows you to search through different Machine Learning
   algorithms and tune each of the algorithm.
-  By searching and tuning ML algorithm to your data you will get very
   accurate model.
-  By calling method ``fit`` from ``Mljar class`` you create new project
   and start experiment with models training. All your results will be
   accessible from your mljar.com account - this makes Machine Learning
   super easy and keeps all your models and results in beautiful order.
   So, you will never miss anything.
-  All computations are done in MLJAR Cloud, they are executed in
   parallel. So after calling ``fit`` method you can switch your
   computer off and MLJAR will do the job for you!
-  I think this is really amazing! What do you think? Please let us know
   at ``contact@mljar.com``.

Examples
--------

The examples are `here! <https://github.com/mljar/mljar-examples>`__.

Testing
-------

To run tests with command:

::

    python -m tests.run

.. |Build Status| image:: https://travis-ci.org/mljar/mljar-api-python.svg?branch=master
   :target: https://travis-ci.org/mljar/mljar-api-python
.. |PyPI version| image:: https://badge.fury.io/py/mljar.svg
   :target: https://badge.fury.io/py/mljar
.. |Coverage Status| image:: https://coveralls.io/repos/github/mljar/mljar-api-python/badge.svg?branch=master
   :target: https://coveralls.io/github/mljar/mljar-api-python?branch=master
