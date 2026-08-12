import math


class Camera:
    """
    Maps world coordinates (metres, y-up) to canvas pixels (y-down).

    The world point (cx, cy) is drawn at the centre of the canvas. Mouse
    interactions update (cx, cy) and scale: pan_screen() shifts by a pixel
    delta and zoom_at() scales while keeping a chosen screen point fixed.
    """

    def __init__(self, canvas_w, canvas_h, scale=40.0, center_world=(0.0, 0.0)):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.scale = scale
        self.cx, self.cy = center_world
        # When True, overlay arrows ignore magnitude and draw at a fixed
        # on-screen length in the vector's direction.
        self.normalize_vectors: bool = False
        self.normalized_pixel_length: float = 60.0

    def w2s(self, wx, wy):
        sx = self.canvas_w * 0.5 + (wx - self.cx) * self.scale
        sy = self.canvas_h * 0.5 - (wy - self.cy) * self.scale
        return sx, sy

    def s2w(self, sx, sy):
        wx = self.cx + (sx - self.canvas_w * 0.5) / self.scale
        wy = self.cy - (sy - self.canvas_h * 0.5) / self.scale
        return wx, wy

    def length(self, metres):
        return metres * self.scale

    def resize(self, canvas_w, canvas_h):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h

    def pan_screen(self, dx, dy):
        # dx/dy are screen-pixel deltas; world moves opposite to the camera
        self.cx -= dx / self.scale
        self.cy += dy / self.scale

    def zoom_at(self, sx, sy, factor):
        # keep the world point currently at (sx, sy) anchored under the cursor
        wx, wy = self.s2w(sx, sy)
        self.scale = max(0.01, self.scale * factor)
        self.cx = wx - (sx - self.canvas_w * 0.5) / self.scale
        self.cy = wy + (sy - self.canvas_h * 0.5) / self.scale

    def arrow_tip(self, base_x, base_y, vec_x, vec_y, default_scale):
        # Screen-space tip of an overlay arrow at world (base_x, base_y) with
        # vector (vec_x, vec_y). Off: length = |vec|/default_scale in world units.
        # On:  length = normalized_pixel_length pixels in vec's direction.
        mag = math.sqrt(vec_x * vec_x + vec_y * vec_y)
        if mag < 1e-9:
            return self.w2s(base_x, base_y)
        if self.normalize_vectors:
            L = self.normalized_pixel_length / self.scale
        else:
            L = mag / default_scale
        ux = vec_x / mag
        uy = vec_y / mag
        return self.w2s(base_x + ux * L, base_y + uy * L)
