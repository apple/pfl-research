.. _fl_introduction:

Federated learning with pfl
===========================

Federated learning (FL) allows training models in a distributed
manner without storing data centrally on a server
(`Konecny et al., 2015 <https://arxiv.org/abs/1511.03575>`_,
`Konecny et al., 2016 <https://arxiv.org/abs/1610.02527>`_).

This section discusses cross-device FL and how it can be implemented
using ``pfl``. The section also provides examples for preparing the
data and the model, which are important inputs to the algorithms
themselves. The section does not provide an exhaustive list
of algorithms implemented in ``pfl`` but rather a few simple examples
to get started.

For a more complete view, the official benchmarks are available in the ``benchmarks``
directory, using a variety of realistic dataset-model combinations with
and without differential privacy.

Cross-device federated learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stochastic gradient descent (SGD) is the standard algorithm
for training neural networks. In a distributed setting, the
training data are split between multiple servers in a data
center that each have a subset of the data, and each server computes the
gradient of the loss function with respect to the model parameters on
its own subset of data.
The sum of the gradients computed by each of the servers is the sum
of the gradients over the union of the data on those servers.
The model parameters are then updated by making a step in the
direction of this gradient.

The federated setting is similar in principle, with a small fraction
of user devices taking the place of the servers in each iteration.
However, in the federated setting the communication links are much
slower, and the data can be unequally distributed amongst devices.
The standard SGD algorithm in this setting is called `federated SGD`.

`Federated averaging <https://arxiv.org/abs/1602.05629>`_ is a
generalized form of federated training. Instead of each device computing
a single gradient, each device performs multiple steps of SGD locally
on its data, and reports the model differences back to the server.
The server then averages the model differences from all
devices in the cohort and uses the average in place of a gradient.
In practice, `adaptive optimizers <https://arxiv.org/abs/2003.00295>`_
are often incorporated into the local or central training.

The number of devices participating in each iteration is referred
to as cohort size (C). C is typically a small fraction of the overall
population of devices.

For practical and privacy reasons, user devices typically cannot maintain a state
across FL rounds, although in some FL algorithms, devices are stateful. It is
often assumed that in practice every user participates in the training at
most once or once in a relatively long period of time.

While FL on its own provides only limited privacy guarantees
(`Boenisch et al., 2023 <https://arxiv.org/abs/2112.02918>`_;
`Carlini et al., 2023 <https://arxiv.org/abs/2202.07646>`_;
`Kariyappa et al., 2023 <https://arxiv.org/abs/2209.05578>`_),
it can be combined with differential privacy (DP)
(`Dwork et al., 2014 <https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf>`_)
and secure aggregation
(`Bonawitz et al., 2016 <http://arxiv. org/abs/1611.04482>`_;
`Talwar et al., 2023 <https://arxiv.org/abs/2307.15017>`_)
to provide strong privacy guarantees for users (or clients) while training
high quality models (`Abadi et al., 2016 <https://arxiv.org/abs/1607.00133>`_)
For example, to incorporate user-level differential privacy using
Gaussian noise, before sending the model differences back to the server, the differences are
first clipped to make sure that the norm is upper bounded by a given clipping
bound, and Gaussian noise is then added to each coordinate. The higher the
noise relative to the clipping bound, the stronger the privacy guarantees.
The clipped and randomized vector is then sent back to the server instead of
the raw model differences.

This document provides a high level example of how to initialize the key
components to get the basic FL simulation running and a few pointers on how
to change these components.

Preparing data
^^^^^^^^^^^^^^
A federated dataset is a collection of smaller datasets that are each associated
to a unique user. The federated dataset can be defined using :class:`.FederatedDataset`, which
takes two key parameters: ``make_dataset_fn`` and ``user_sampler``. We discuss these two parameters next.

The first parameter, ``make_dataset_fn``, is a function that returns the data
of a particular user given the user ID. This is the place where you want to do any preprocessing.
For example, imagine that there is one file that represents the data from each user:

.. code-block:: console

    $ cat user1.json
    {"x": [[0, 0], [1, 0], [0, 1], [1, 1]], "y": [0, 0, 0, 1]}

The data loading function in this case can be implemented as follows:

.. code-block:: python

    from pfl.data.dataset import Dataset

    def make_dataset_fn(user_id):
        data = json.load(open('{}.json'.format(user_id), 'r'))
        features = np.array(data['x'])
        labels = np.eye(2)[data['y']] # Make one-hot encodings
        return Dataset(raw_data=[features, labels])

In the above example, the raw data of the returned ``Dataset`` is a list of two entries. The first entry is the ``x`` argument and the second entry is the ``y`` argument. These arguments must match the ``loss`` and ``metric`` functions of the model.

The expected order of the data inputs for other deep learning frameworks is described in their corresponding :ref:`models`.

The second parameter of :class:`~pfl.data.federated_dataset.FederatedDataset`, ``user_sampler``, should also be a callable, and will return a sampled user identifier every call.
``pfl`` implements two different sampling functions by default (available from the factory function :func:`~pfl.data.sampling.get_user_sampler`): random and minimize reuse.
Random sampling generates each cohort with a uniform distribution.
The minimize-reuse sampler maximizes the time between instances of reuse of the same user (see :class:`~pfl.data.sampling.MinimizeReuseUserSampler`).

Although the random user sampler might seem the obvious choice because the cohorts in live FL deployments are typically
selected at random, with a limited number of users available for the FL simulation, the minimize-reuse sampling may in fact have a more realistic behavior.

.. code-block:: python

    >>> from pfl.data.sampling import get_user_sampler
    >>> user_ids = ['user1', 'user2', 'user3']
    >>> sampler = get_user_sampler('minimize_reuse', user_ids)
    >>> for _ in range(5):
    >>>    print('sampled ', sampler())
    'sampled user1'
    'sampled user2'
    'sampled user3'
    'sampled user1'
    'sampled user2'

When you have defined a callable for the parameter ``make_dataset_fn`` and a callable for the parameter ``user_sampler``, the federated dataset can be created.

.. code-block:: python

    dataset = FederatedDataset(make_dataset_fn, sampler)


The dataset can be iterated through, sampling a user dataset each call.

.. code-block:: python

    >>> next(dataset).raw_data
    [array([[0, 0],
            [1, 0],
            [0, 1],
            [1, 1]]),
     array([[1., 0.],
            [1., 0.],
            [1., 0.],
            [0., 1.]])]


For more information on how to prepare datasets and federated datasets,
please see the tutorial in TODO and benchmarks in TODO.

Defining a model
^^^^^^^^^^^^^^^^

Below we define a simple PyTorch model that can be used for binary classification with
10 input features, and it includes binary cross-entropy loss and accuracy metrics. Note that the
``loss`` and ``metrics`` functions have two arguments, ``x`` and ``y``, which we discussed above
when defining the dataset.

.. code-block:: python

    import torch
    from pfl.model.pytorch import PyTorchModel

    class TestModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)
            self.activation = torch.nn.Sigmoid()

        def forward(self, x):  # pylint: disable=arguments-differ
            x = self.linear(x)
            x = self.activation(x)
            return x

        def loss(self, x, y, eval=False):
            self.eval() if eval else self.train()
            bce_loss = torch.nn.BCELoss(reduction='sum')
            return bce_loss(self(torch.FloatTensor(x)), torch.FloatTensor(y))

        def metrics(self, x, y):
            loss_value = self.loss(x, y, eval=True)
            num_samples = len(y)
            correct = ((self(x) > 0.5) == y).float().sum()
            return {
                'loss': Weighted(loss_value, num_samples),
                'accuracy': Weighted(correct, num_samples)
            }

    pytorch_model = TestModel()
    model = PyTorchModel(model=pytorch_model,
                         local_optimizer_create=torch.optim.SGD,
                         central_optimizer=torch.optim.SGD(
                             pytorch_model.parameters(), lr=1.0))

FL algorithms in pfl
^^^^^^^^^^^^^^^^^^^^

Federated averaging
"""""""""""""""""""
To implement cross-device FL with federated averaging using ``pfl``, the key algorithm to use is
:class:`.FederatedAveraging`:

.. code-block:: python

  from pfl.algorithm.federated_averaging import FederatedAveraging

  algorithm = FederatedAveraging()

Assuming we want to train a neural network, we can proceed by setting the key
parameters for central and local training, and evaluation:

.. code-block:: python

  algorithm_params = NNAlgorithmParams(
        central_num_iterations=central_num_epochs,
        evaluation_frequency=10,
        train_cohort_size=cohort_size,
        val_cohort_size=val_cohort_size)

    model_train_params = NNTrainHyperParams(
        local_num_epochs=local_num_epochs,
        local_learning_rate=local_learning_rate,
        local_batch_size=None)

    model_eval_params = NNEvalHyperParams(local_batch_size=None)

Backend simulates an algorithm on the given federated dataset, which
includes sampling the users, running local training, applying
privacy mechanisms and applying postprocessors:

.. code-block:: python

    backend = SimulatedBackend(training_data=dataset,
                               val_data=val_dataset,
                               postprocessors=[])

Callbacks can be provided that can be run at various stages of
the algorithm. In the example shown below, the callbacks enable
evaluating the model on the central dataset before the training begins
and between central iterations, and saving aggregate metrics after
each 100 iterations:

.. code-block:: python

    cb_eval = CentralEvaluationCallback(central_dataset,
                                        model_eval_params)

    cb_save = AggregateMetricsToDisk(
        output_path=output_path,
        frequency=100,
        check_existing_file=False,
    )

The algorithm can then be run:

.. code-block:: python

    algorithm.run(
        backend=backend,
        model=model,
        algorithm_params=algorithm_params,
        model_train_params=model_train_params,
        model_eval_params=model_eval_params,
        callbacks=[cb_eval, cb_save])

.. _Reptile-example:

Reptile: FL with fine-tuning (personalization)
""""""""""""""""""""""""""""""""""""""""""""""

:class:`.Reptile`
(`Nichol et al., 2018 <https://arxiv.org/abs/1803.02999>`_)
combines federated averaging with fine-tuning where the
model is fine tuned locally on each device prior to evaluation. Therefore,
compared to traditional federated averaging, the evaluation should focus
on metrics after running the local training. It is straightforward to switch
the algorithm to enable fine-tuning (using the same parameters as in federated
averaging):

.. code-block:: python

    from pfl.algorithm.reptile import Reptile

    reptile = Reptile()

    reptile.run(
        backend=backend,
        model=model,
        algorithm_params=algorithm_params,
        model_train_params=model_train_params,
        model_eval_params=model_eval_params,
        callbacks=[cb_eval, cb_save])


.. _GBDT-example:

Gradient Boosted Decision Trees
"""""""""""""""""""""""""""""""

This section presents an example of using ``pfl`` to train a gradient boosted
decision tree (GBDT) model with a
specialized training algorithm. In this case, the algorithm incrementally
grows the trees.

The parameters for GBDT algorithm are defined using :class:`.GBDTAlgorithmHyperParams`:

.. code-block:: python

    from pfl.tree.federated_gbdt import GBDTAlgorithmHyperParams
    from pfl.tree.gbdt_model import GBDTModelHyperParams

    gbdt_algorithm_params = GBDTAlgorithmHyperParams(
        cohort_size=cohort_size,
        val_cohort_size=val_cohort_size,
        num_trees=20)
    model_train_params = GBDTModelHyperParams()
    model_eval_params = GBDTModelHyperParams()


Two versions of GBDT models are implemented:
:class:`.GBDTModelClassifier` implements GBDT for classification and
:class:`.GBDTModelRegressor` implements GBDT for regression. Here is
an example of creating a GBDT classifier model:

.. code-block:: python

    from pfl.tree.gbdt_model import GBDTModelClassifier

    model = GBDTModelClassifier(num_features=num_features, max_depth=3)

To initialize the GBDT training algorithm, it's necessary to provide details
about the features. The code snippet below provides an example with 100 bool
features and 10 floating point features from interval [0, 100] with 5
equidistant boundaries to consider for tree splits:

.. code-block:: python

    from pfl.tree.tree_utils import Feature

    features = []
    for i in range(100):
        features.append(Feature(2, (0, 1), bool, 1))
    for i in range(10):
        features.append(Feature(1, (0, 100), float, 5, 'equidistant')

    gbdt_algorithm = FederatedGBDT(features=features)

The algorithm can then be run similarly as in other examples:

.. code-block:: python

    gbdt_algorithm.run(algorithm_params=gbdt_algorithm_params,
                       backend=backend,
                       model=model,
                       model_train_params=model_train_params,
                       model_eval_params=model_eval_params,
                       callbacks=[cb_eval, cb_save])


Implementing new FL algorithms in pfl
"""""""""""""""""""""""""""""""""""""

The above examples provide good starting points on how to implement
new FL algorithms, although simpler versions can often be created
by focusing on a single framework.

Most new algorithms are likely
to extend :class:`.FederatedAveraging`.
If the new algorithm requires
the users to store states, consider using :class:`.SCAFFOLD` as an example
of how to initialize and update user states. If the new algorithm
modifies the loss function (e.g. by adding a regularization term),
:class:`.FedProx` is a good starting point.
If the algorithm modifies the training loop in some way, :ref:`Reptile-example`
provides a good example. Finally, :ref:`GBDT-example`
provide examples of implementing algorithms that require specialized
training and evaluation instead of the typical federated averaging.

From FL to PFL: Incorporating Privacy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We discussed above that FL on its own does not guarantee privacy, and
that is why we may want to incorporate differential privacy (DP) into FL.
Private federated learning (PFL) is simply FL with
DP, which can in practice be combined with secure aggregation.
For more information on how to do incorporate DP into FL
simulations using ``pfl``, please see TODO and benchmarks
in folder ``benchmarks/`` of this repository.