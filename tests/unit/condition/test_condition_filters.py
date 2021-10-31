from perambulate import Condition


def test_touches(sinusoid_d):
    A = Condition(condition=(sinusoid_d > -0.5) & (sinusoid_d < 0.5))
    B = Condition(condition=(sinusoid_d > -1) & (sinusoid_d < 1))

    C = A.touches(B)

    assert C == A
    assert C != B.touches(A)


def test_encloses(sinusoid_d):
    A = Condition(condition=(sinusoid_d > -0.5) & (sinusoid_d < 0.5))
    B = Condition(condition=(sinusoid_d > -1) & (sinusoid_d < 1))

    assert B.encloses(A) == B
    assert len(A.encloses(B)) == 0


def test_inside(sinusoid_d):
    A = Condition(condition=(sinusoid_d > -0.5) & (sinusoid_d < 0.5))
    B = Condition(condition=(sinusoid_d > -1) & (sinusoid_d < 1))

    assert A.inside(B) == A
    assert len(B.inside(A)) == 0
