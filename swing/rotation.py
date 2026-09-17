import math

from pose.detector import Landmark

def rotation_from_span(span: float | None, address_span: float | None) -> float | None:
    """Rotation angle from how much a projected line has shrunk since address.

    A rigid segment (the hip or shoulder line) rotating in 3D projects onto
    the camera's x-axis as span = true_length * cos(theta), where theta is
    the rotation away from the address pose. Treating address_span as a
    stand-in for true_length gives theta = acos(span / address_span) -- a
    weak-perspective approximation, not a measurement of true 3D rotation.

    It assumes the segment's real length and its distance to the camera stay
    constant through the swing: lateral weight shift, or moving toward/away
    from the camera, both violate that and bias the estimate. acos's
    derivative is also steep near ratio ~= 1.0, so ordinary landmark jitter
    at address/early-backswing (where the ratio starts near 1) gets amplified
    into angle noise.
    """
    if span is None or address_span is None or address_span < 0.001:
        return None
    ratio = max(0.0, min(1.0, span / address_span))
    return math.degrees(math.acos(ratio))

def _line_angle_xz(a: Landmark, b: Landmark) -> float:
    """Angle in degrees of the line a->b projected onto the x/z plane (bird's-eye view)."""
    return math.degrees(math.atan2(b.z - a.z, b.x - a.x))

def rotation_from_depth(a: Landmark, b: Landmark, address_a: Landmark, address_b: Landmark) -> float:
    """Rotation estimate from MediaPipe's own z coordinate: how much the line a->b has
    turned in the x/z plane since address. This is a self-consistency check against
    rotation_from_span, not ground truth -- z is itself network-estimated depth, not a
    measured quantity. Returned as an unsigned angle in [0, 180] to match acos's range."""
    diff = abs(_line_angle_xz(a, b) - _line_angle_xz(address_a, address_b))
    return diff if diff <= 180 else 360 - diff
