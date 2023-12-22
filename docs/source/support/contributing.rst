.. _Contributing:

Contributing
============

We welcome contributions from anyone who wish to improve ``pfl`` for research.
The framework is maintained in `pfl-research Git repository`_ and anything related to benchmarking and new research goes into `pfl-research Git repository`_.
Below are guidelines on how to make contributions to ``pfl`` that follow our best practices.
Reach out to us on ``fgranqvist@apple.com`` if you have any questions about developing on ``pfl`` and see `open issues <https://github.com/apple/pfl-research/issues>`_ for community reported bugs and beginner friendly tasks.

Setting up development environment
----------------------------------

The ``main`` branch always contains the latest release of ``pfl``.
Any new features and improvements should be based off the development branch called ``develop``.
If you would like to contribute to ``pfl``, a first step is to set up your development environment by following these steps:

1. Create a fork of ``pfl-research`` repo using the github UI: go to `pfl-research Git repository`_ and click on ``Fork``. **Do not keep your branches in the main pfl-research repo**.
2. Clone the forked repository to your local computer.
3. Execute these commands:

.. code-block:: bash

    # Add apple/pfl-research as an upstream
    git remote add upstream https://github.com/apple/pfl-research
    git fetch upstream

    # Checkout a new feature branch, called 'new-awesome-feature', based off "develop" from upstream
    git checkout upstream/develop -b new-awesome-feature

Prerequisite - you need to have `Poetry`_ installed:

.. code-block:: bash

   curl -sSL https://install.python-poetry.org | python3 -

Prerequisite - if you don't have the correct Python version installed which is constrainted in `pyproject.toml`_, install it with conda. You only need to have this activated the first time you install the poetry environment. 

.. code-block:: bash

   conda create -n py310 python=3.10
   conda activate py310


Install environment:

.. code-block:: bash

    # If you have the new Python 3.10 environment active, it should be cloned.
    poetry env use `which python`
    # Install with whatever extras and dev groups you need for development.
    poetry install -E pytorch -E tf -E trees --with dev,docs
    # Activate environment
    poetry shell

This will install ``pfl`` in a reproducible environment identical to everybody working on ``pfl``, including our CI build pipelines.

Compiling the documentation
---------------------------

If you have improvements for the documentation, or if you add things that require a change in the documentation, we look forward to finding this in a pull request to the ``develop`` branch.

The documentation is built with Sphinx.
To build the documentation, you need to install ``pfl`` at minimum with the ``docs`` dependency group: 

.. code-block:: bash

    poetry install --only docs

Once all required packages are installed, building is easy:

.. code-block:: bash

    make docs

Or use the underlying poetry command:

.. code-block:: bash

    poetry run make -C docs html

Any PR that adds a public interface should have a docstring with `Sphinx-style docstring formatting`_ and be imported into the reference documentation at ``doc/source/reference``.


Contributing to code
--------------------

Development process
~~~~~~~~~~~~~~~~~~~

We have a few rules with regards to the process of developing a new feature in ``pfl``.
We follow `semantic versioning <https://semver.org/>`_.
This means that we do not accept any new changes into pfl's current major version that cause a breaking change to the public API.
See :ref:`code structure <code_structure>` for information about what parts of ``pfl`` are classified as public API.

We don't have an RFC process in place yet for external contributions.
If you wish to implement a new feature that will require at least a minor version bump according to semantic versioning we suggest that you first open an `issue <https://github.com/apple/pfl-research/issues>`_ and explain the why, what and how and get some initial feedback.

There are 2 active branches in the main repository (and 1 additional when preparing to release):

* **main** - This is the current released version of ``pfl``, which is also available on PyPI. The only branches that should be merged into this branch are release branches.
* **develop** - This is the working branch to base from when developing a new feature and PRs should be directed toward this branch as long as they are compatible with the current major version of ``pfl``.
* **release-x.y.z** - These branches are made when we prepare for a release, and are merge commited into *main*.

Feature PRs merged into **develop** and **release-x.y.z** should be squash merged.
Merging **release-x.y.z → main and main → develop** should be regular merge commits such that the history is kept.


Standardizing the code
~~~~~~~~~~~~~~~~~~~~~~

We like to keep the quality of the code high and reduce the ambiguity in how it should be formatted.
For that, we use `yapf`_, `ruff`_  and `mypy`_.

Following `Google’s Python styleguide <https://google.github.io/styleguide/pyguide.html>`_ is preferred, but it is acceptable to deviate from it on your own discretion because we do not strictly require it.
Unlike Google’s styleguide, we use `Sphinx-style docstring formatting`_.
Do not include `:type` and `:rtype`, since that is replaced by type hints.
For consistency, code should be formatted with `yapf`_ of the version specified in `pyproject.toml`_.
`yapf`_ automates a subset of the rules mentioned in the styleguide.
To run `yapf`_, integrate it in your IDE or manually run the following command in the root directory of ``pfl-research``:

.. code-block:: bash

    make yapf

Or use the underlying poetry command:

.. code-block:: bash

    poetry run yapf -i --recursive --parallel pfl/ tests/

``pfl`` uses `type hints`_ to ensure the design is coherent and to provide valuable documentation about parameters of interfaces.
We use `mypy`_ for the type checking itself.

.. code-block:: bash

    poetry run mypy

We also use `ruff`_ for consistency.
The settings to use for linting ``pfl`` is included in `pyproject.toml`_ in the root directory.

You can integrate `ruff`_ with your IDE or manually run the following command to check for linting issues in all of ``pfl`` and perform type checking.

.. code-block:: bash

    poetry run ruff check --diff pfl/ tests/

In summary, the checks your branch needs to pass to be able to contribute the changes are `pytest`_, `yapf`_, `mypy`_, `ruff`_ and successfully compiling the documentation.
Don't worry if you might have missed any of these, there is a CI build pipeline for each check that will run when you submit your PR.

Testing
~~~~~~~

We like making sure that your code works and keeps working even if we make changes.
Therefore, we have unit tests for anything that is reasonably unit-testable.
To be merged, the code in the pull request should pass all tests.
To run all tests locally, install `pytest`_ and run it in the root dir:

.. code-block:: bash

   make test

Or use the underlying poetry command:

.. code-block:: bash

	poetry run pytest -svx

To test all different environments ``pfl`` is expected to be compatible with,
install `tox`_ and run ``tox`` in the root dir.

Tests are all located in the ``tests`` directory.
Larger functional/integration tests are located in ``tests/integration`` and are currently only required for features related to multi-worker and multi-process simulations.
New unit tests should be placed in a file in ``tests`` which mirrors the path of the feature in the ``pfl`` module.
The class and method names should also be mirrored. Example:

.. code-block:: python

    # pfl/model/my_model.py
    class MyModel:
        def my_method(self):
            # ... implementation

    # tests/model/test_my_model.py
    class TestMyModel:
        def test_my_method(self):
            # ... test implementation

Try to re-use existing mocked components from the ``conftest.py`` files before you decide to create a new mocked component.

If your test will only be able to run on MacOS (currently, CI build pipelines to not support MacOS), use the decorator ``@pytest.mark.macos``.
The test will then only be enabled when run with:

.. code-block:: bash

	poetry run pytest --macos
            

Package dependencies
~~~~~~~~~~~~~~~~~~~~

It is not trivial to manage a package which supports use-cases using TF, PyTorch or none of these packages.
We keep dependencies of different deep learning frameworks separate with install extras, see `pyproject.toml`_.
If your feature has a new dependency but it is only relevant when used in combination with a particular deep learning framework, then put the dependency in the appropriate install extra.
In `pyproject.toml`_, constrain the new dependency from the earliest known **minor** version, up to but not including the next **major** version.
Thereafter, update the lock file to update pinned versions for local development and CI:

.. code-block:: bash

   poetry lock --no-update

This will ensure both developers and CI have reproducible builds.

.. _code_structure:

Code structure
~~~~~~~~~~~~~~

.. note::

   When implementing components for your own research without the plan to contribute back to ``pfl``, you can of course import e.g. ``torch`` anywhere you want and don't have to adhere to these rules.

Everything in ``pfl.internal`` is **not** considered a part of the public API, everything else in the ``pfl`` module is.
This means that you are free to make breaking interface changes if the code is located in ``pfl.internal``, but not to other components because of semantic versioning.
Users should be able to run ``pfl`` with 0 or 1 deep learning frameworks without the need to install all deep learning frameworks.
For this we keep the TF and PyTorch code encapsulated in a few modules, which should not be automatically imported into ``__init__.py`` files.
Code for each deep learning framework should be encapsulated in 1 dataset module, 1 model module, 1 ops module and 1 bridge module.

.. note::
    For example, all PyTorch code is encapsulated in:

    * ``pfl.data.pytorch`` - PyTorch native dataset support
    * ``pfl.model.pytorch`` - PyTorch models
    * ``pfl.internal.ops.pytorch_ops`` - low-level PyTorch-specific operations. This module does not depend on any other components in ``pfl``.
    * ``pfl.internal.bridge.pytorch`` - Concrete PyTorch implementations of interfaces that are used to inject framework-specific code into algorithms. ``pfl`` primitives such as :class:`~pfl.metrics.Metrics` or :class:`~pfl.stats.TrainingStatistics` may be used here.

To inject code from the particular deep learning framework currently in use, put it in the ops module if it is a single function that has no other dependencies, or make a new interface in ``pfl.internal.bridge`` if it is a collection of functions that belong together to make a certain feature work and is optional to implement for any one deep learning framework .

How to use the current selected ops module:

.. code-block:: python

    from pfl.internal.ops.selector import get_default_framework_module as ops
    ops().get_shape(tensor)


Making a pull request
~~~~~~~~~~~~~~~~~~~~~

Once the code is standardized and tests are implemented in the ``tests`` directory, you are ready to do a pull request.
For that you just need to push the files you modified to the branch of your forked repository.

.. code-block:: bash 

    git add files-you-changed
    git commit -m 'Message explaining your changes'
    git push origin new-awesome-feature

Finally, you can go do a pull request from the Github UI: go to ``https://github.com/apple/pfl-research/compare`` and click on ``Create pull request``. Then, make sure that you reference any open issues that the PR solves and leave an informative message about the changes made in the ``new-awesome-feature`` branch.

The next step is to wait for the CI builds to pass (progress is shown at the bottom of the PR).
Contributors with admin status may have to kickstart the CI for you.
Contact us by mentioning Apple team members in the PR or reach out to ``fgranqvist@apple.com``.

**Checklist for a PR to pass review**

This is a checklist for the reviewer to go through in addition to reviewing the code quality.
The contributor should take this checklist into account to ensure a smooth and quick process for getting the PR merged.

1. Corresponding documentation should also be updated: reference docstring and tutorials.
2. Unit tests should have good coverage: positive test (should pass), negative test (should fail), test for numerical stability if relevant, statistical test if relevant.
3. If relevant, add test case to any integration test this PR can affect (in ``tests/integration``).
4. Check if `pfl-research Git repository`_ repo should be updated along with this PR.
5. Pass all PR builds (pytest, yapf, ruff, mypy, build wheel, build docs). This is run by a pfl admin on the contributor’s behalf.
6. A small description of the change is included in CHANGELOG.md if it is relevant to notify users about it.

.. _yapf: https://github.com/google/yapf
.. _pytest: https://docs.pytest.org/en/stable/
.. _tox: https://tox.readthedocs.io/en/latest/
.. _ruff: https://docs.astral.sh/ruff/
.. _mypy: https://mypy-lang.org/
.. _type hints: https://docs.python.org/3.7/library/typing.html
.. _Sphinx-style docstring formatting: https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html
.. _pfl Git repository: https://github.com/apple/pfl
.. _pfl-research Git repository: https://github.com/apple/pfl-research
.. _Poetry: https://python-poetry.org
.. _pyproject.toml: https://github.com/apple/pfl-research/blob/main/pyproject.toml

