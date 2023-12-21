#!/bin/bash

# This returns non-0 error code even when succeeds.
# No information found online about bug.
# May be caused by using the torch_cu118 source.
poetry install --with dev -E pytorch -E trees

if [[ $? -ne 0 ]]; then
  echo "Return code was ${?}, but ignore it"
fi
