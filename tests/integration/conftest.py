# Copyright © 2023-2024 Apple Inc.
import itertools
import os
import signal
import subprocess

import pytest


@pytest.fixture(scope='function')
def ports(request):
    """
    Fixture for choosing the correct ports for pfl's multiworker debug mode.

    Pytest runs the tests in parallel, which means different ports needs to be
    chosen each time the test is called with a specific combination of
    parametrized parameters. Make sure the ports chosen is unique with respect
    to the values of the parametrized input parameters.
    """

    all_params_all_values = []
    current_param_combination = []

    def maybe_get_id(obj):
        """
        lazy_fixture is not hashable.
        Therefore, always try to get the id of a parameter from
        pytest.mark.parametrize as a first option.
        """
        try:
            return obj.id
        except:
            return obj

    # Find all combinations of arguments for the test method.
    for mark in request.keywords.get('pytestmark'):
        comma_separates_param_names, params_values = mark.args

        param_names = comma_separates_param_names.split(',')

        if len(param_names) > 1:
            # params_values should be a list of tuples.
            for i, param_name in enumerate(param_names):

                current_param_combination.append(
                    request.getfixturevalue(param_name))
                one_param_values = [
                    maybe_get_id(tup[i]) for tup in params_values
                ]
                all_params_all_values.append(one_param_values)
        else:
            current_param_combination.append(
                maybe_get_id(request.getfixturevalue(param_names[0])))
            all_params_all_values.append(
                [maybe_get_id(v) for v in params_values])

    all_params_combinations = sorted(
        set(itertools.product(*all_params_all_values)))

    unique_param_combination_number = all_params_combinations.index(
        tuple(current_param_combination))

    # When running tox in parallel, we need to avoid any race conditions with
    # the port numbers.
    # Simply use a different port range for each env in tox by assuming each env
    # has an environment variable TOX_ENV_ID with a unique integer value > 0
    tox_multiplier = int(
        os.environ['TOX_ENV_ID']) if 'TOX_ENV_ID' in os.environ else 1

    # These ports should be unique.
    port1 = 8050 + tox_multiplier * len(
        all_params_combinations) * 2 + unique_param_combination_number * 2
    port2 = 8050 + tox_multiplier * len(
        all_params_combinations) * 2 + unique_param_combination_number * 2 + 1
    yield port1, port2

    # After all tests, force kill any remaining TensorFlow servers.
    # Otherwise, they will remain up and block the ports.

    # These are the debug ports.
    for port in [port1, port2]:
        process = subprocess.Popen(["lsof", "-i", f":{port}"],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, _ = process.communicate()
        for process in str(stdout.decode("utf-8")).split("\n")[1:]:
            data = [x for x in process.split(" ") if x != '']
            if len(data) <= 1:
                continue

            # Kill process listening on port.
            os.kill(int(data[1]), signal.SIGKILL)
