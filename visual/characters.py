"""The three brothers: Theo (Transformer), Manu (Mamba), Max (Mamba-2)."""

from manim import DOWN, PI, RIGHT, UP, Arc, Circle, Dot, Line, RoundedRectangle, Square, VGroup

from .theme import BG1, K_COL, MAMBA, Q_COL, RESID, V_COL, T


def brother(name: str, color, prop: str, scale: float = 1.0) -> VGroup:
    """A small character holding a prop: 'cards' (Theo), 'notebook' (Manu) or 'tiles' (Max).

    Returns VGroup(figure, prop, name_label).
    """
    head = Circle(radius=0.36, stroke_width=0, fill_color="#F6E3D2", fill_opacity=1)
    hair = Arc(radius=0.36, start_angle=0.15, angle=PI - 0.3, stroke_color=color, stroke_width=9)
    eyes = VGroup(*[Dot(radius=0.045, color=BG1).move_to(head.get_center() + RIGHT * dx + UP * 0.03) for dx in (-0.13, 0.13)])
    smile = Arc(radius=0.14, start_angle=PI * 1.15, angle=PI * 0.7, stroke_color=BG1, stroke_width=3)
    smile.move_to(head.get_center() + DOWN * 0.1)
    body = RoundedRectangle(width=1.05, height=0.95, corner_radius=0.32, stroke_width=0, fill_color=color, fill_opacity=0.95)
    body.next_to(head, DOWN, buff=-0.06)
    body.set_z_index(head.z_index - 1)
    if prop == "cards":
        p = VGroup(*[RoundedRectangle(width=0.3, height=0.42, corner_radius=0.05, stroke_color=BG1, stroke_width=1.5,
                                      fill_color=c, fill_opacity=1).rotate(a)
                     for c, a in zip((K_COL, V_COL, Q_COL, K_COL), (0.45, 0.15, -0.15, -0.45))]).arrange(RIGHT, buff=-0.12)
    elif prop == "notebook":
        p = VGroup(RoundedRectangle(width=0.7, height=0.5, corner_radius=0.06, stroke_color=BG1, stroke_width=1.5,
                                    fill_color=RESID, fill_opacity=1),
                   *[Line([-0.22, 0, 0], [0.22, 0, 0], stroke_width=2, color=MAMBA).shift(UP * (0.12 - 0.12 * i)) for i in range(3)])
        p[1:].move_to(p[0])
    else:
        p = VGroup(*[Square(0.17, stroke_color=BG1, stroke_width=1, fill_color=RESID, fill_opacity=0.95)
                     for _ in range(6)]).arrange_in_grid(2, 3, buff=0.04)
    p.move_to(body.get_center() + DOWN * 0.1).set_z_index(5)
    lab = T(name, 24, color, bold=True).next_to(body, DOWN, buff=0.18)
    return VGroup(VGroup(body, head, hair, eyes, smile), p, lab).scale(scale)
