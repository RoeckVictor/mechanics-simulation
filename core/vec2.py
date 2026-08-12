import math


class Vec2:
    """Immutable-style 2D vector. All operations return new instances"""

    __slots__ = ("x", "y")

    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        return Vec2(self.x / scalar, self.y / scalar)

    def __neg__(self):
        return Vec2(-self.x, -self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f"Vec2({self.x:.4f}, {self.y:.4f})"

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def cross(self, other):
        """Scalar z-component of the 3D cross product."""
        return self.x * other.y - self.y * other.x

    def magnitude(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def magnitude_sq(self):
        return self.x * self.x + self.y * self.y

    def normalized(self):
        m = self.magnitude()
        return Vec2(self.x / m, self.y / m) if m > 1e-12 else Vec2(0.0, 0.0)

    def rotate(self, angle):
        c, s = math.cos(angle), math.sin(angle)
        return Vec2(self.x * c - self.y * s, self.x * s + self.y * c)

    def angle(self):
        """Angle from positive x-axis in radians."""
        return math.atan2(self.y, self.x)

    def perp(self):
        """90° CCW perpendicular."""
        return Vec2(-self.y, self.x)

    @staticmethod
    def from_polar(r, theta):
        return Vec2(r * math.cos(theta), r * math.sin(theta))

    @staticmethod
    def zero():
        return Vec2(0.0, 0.0)
