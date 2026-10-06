# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Image-hook stand-in for the check tests."""


def reject(submission, config) -> str:
    return "rejected by hook"
