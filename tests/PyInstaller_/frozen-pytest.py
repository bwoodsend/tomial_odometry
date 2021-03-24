# -*- coding: utf-8 -*-
"""
Freeze pytest.main() with tomial_odometry included.
"""

import sys
import tomial_odometry

import pytest

sys.exit(pytest.main(sys.argv[1:] + ["--no-cov", "--tb=native"]))
