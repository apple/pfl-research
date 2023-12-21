#!/bin/bash

# Copyright © 2023-2024 Apple Inc.

# Build the documentation and push the new version to gh-pages.
# Call this from the root directory.

set -o nounset -o pipefail -o errexit
# Exit if any command fails
set -ex

# Step 1: Build the documentation
poetry install --only docs
poetry run make -C docs html

# Step 2: Publish to GitHub Pages
# Copy the built documentation to a temporary directory
mkdir -p /tmp/docs
cp -R docs/build/html/* /tmp/docs/

git checkout -B gh-pages

# Copy the built docs from the temporary directory
cp -R /tmp/docs/* .

# https://docs.github.com/en/pages/getting-started-with-github-pages/about-github-pages#static-site-generators
touch .nojekyll

# Push to GitHub
git add -A
git commit -m "Update documentation"
git push upstream gh-pages -f

# Switch back to the original branch
git checkout -

# Clean up
rm -rf /tmp/docs

