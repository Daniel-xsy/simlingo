#!/usr/bin/env python3
"""
Backward-compatible re-export shim.

Existing consumers that ``from language_navigation.utils import X`` can
instead ``from language_navigation_dev.utils import X`` and get the same
names.  For new code, import directly from the focused submodules:

    * ``geometry`` — distance / position / lane helpers
    * ``opendrive`` — speed-limit resolver, CARLA map cache
    * ``instructions`` — instruction library, text sampling, accel helpers
    * ``actionability`` — turn scanning, trigger selection, special cases
    * ``xml_builder`` — XML indent, evaluation, scenarios, route tree
"""

from language_navigation.opendrive import *      # noqa: F401,F403
from language_navigation.geometry import *       # noqa: F401,F403
from language_navigation.actionability import *  # noqa: F401,F403
from language_navigation.instructions import *   # noqa: F401,F403
from language_navigation.xml_builder import *    # noqa: F401,F403
