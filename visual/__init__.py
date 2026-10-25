"""Deck toolkit. Slides do `from visual import *` to get Manim plus everything below.

    theme       palette, fonts, T() text, M() plain maths
    equations   ceq() colour-coded maths, morph() one substitution step, collapse() steps -> result
    shapes      card, pill, chip, token_card, heat, triangle, sheet, bars, glow, arrow, ...
    characters  brother()
    deck        Deck / Deck3D scene base classes
"""

from manim import *  # noqa: F401,F403
import numpy as np  # noqa: F401

from .theme import *  # noqa: F401,F403
from .equations import boxed, ceq, collapse, morph, residual_eq, role, settle  # noqa: F401
from .shapes import (arrow, bars, card, causal_softmax, chip, dial, glow, heat, hourglass, pill,  # noqa: F401
                     sheet, token_card, triangle, vec)
from .characters import brother  # noqa: F401
from .deck import Deck, Deck3D  # noqa: F401
