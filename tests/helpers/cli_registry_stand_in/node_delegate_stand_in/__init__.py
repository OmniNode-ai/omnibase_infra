# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A stand-in delegate node package: a ``contract.yaml`` and its input model.

Registered in ``onex.nodes`` by :func:`install_stand_in_registry` under the
delegate node's name, so ``onex delegate`` reads this contract's input model
exactly as it reads the real one. Its vocabularies are this fixture's own and
are not copied from any production contract; tests assert that the CLI offers
whatever this declares.
"""
