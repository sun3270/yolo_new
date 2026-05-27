# ---- LGMSF-Lite modules: begin ----
class ConvBNAct(nn.Module):
    """Small Conv-BN-activation helper used by LGMSF-Lite."""

    def __init__(self, c1, c2, k=1, s=1, p=None, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWSeparableConv(nn.Module):
    """Depthwise separable convolution."""

    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        self.dw = ConvBNAct(c1, c1, k=k, s=s, groups=c1, act=act)
        self.pw = ConvBNAct(c1, c2, k=1, s=1, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))


class LDSConv(nn.Module):
    """Learnable depthwise-separable downsample convolution."""

    def __init__(self, c1, c2, k=3, s=2, act=True):
        super().__init__()
        self.conv = DWSeparableConv(c1, c2, k=k, s=s, act=act)

    def forward(self, x):
        return self.conv(x)


class GhostConvLite(nn.Module):
    """Lightweight Ghost-style convolution without external dependencies."""

    def __init__(self, c1, c2, k=1, s=1, ratio=2, act=True):
        super().__init__()
        c_primary = max(1, int((c2 + ratio - 1) // ratio))
        c_cheap = c2 - c_primary
        self.primary = ConvBNAct(c1, c_primary, k=k, s=s, act=act)
        self.cheap = ConvBNAct(c_primary, c_cheap, k=3, s=1, groups=c_primary, act=act) if c_cheap > 0 else None
        self.c2 = c2

    def forward(self, x):
        y = self.primary(x)
        if self.cheap is None:
            return y
        return torch.cat((y, self.cheap(y)), dim=1)[:, : self.c2]


class TextureBranch(nn.Module):
    """P3 texture stream for small disease spots and edge detail."""

    def __init__(self, c1, c2):
        super().__init__()
        self.proj = ConvBNAct(c1, c2, k=1, s=1) if c1 != c2 else nn.Identity()
        self.conv1 = DWSeparableConv(c2, c2, k=3, s=1)
        self.conv2 = DWSeparableConv(c2, c2, k=3, s=1)

    def forward(self, x):
        x = self.proj(x)
        return x + self.conv2(self.conv1(x))


class SemanticBranch(nn.Module):
    """P5 semantic stream with low-cost receptive field expansion."""

    def __init__(self, c1, c2):
        super().__init__()
        self.proj = ConvBNAct(c1, c2, k=1, s=1) if c1 != c2 else nn.Identity()
        self.conv = nn.Sequential(
            GhostConvLite(c2, c2, k=1, s=1),
            ConvBNAct(c2, c2, k=5, s=1, groups=c2),
            GhostConvLite(c2, c2, k=1, s=1),
        )

    def forward(self, x):
        x = self.proj(x)
        return x + self.conv(x)


class FastNormFuse2(nn.Module):
    """BiFPN-style fast normalized weighted fusion for two inputs."""

    def __init__(self, eps=1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.eps = eps

    def forward(self, x1, x2):
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        return w[0] * x1 + w[1] * x2


class SimAM(nn.Module):
    """Parameter-free SimAM attention."""

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        _b, _c, h, w = x.size()
        n = h * w - 1
        if n <= 0:
            return x
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * torch.sigmoid(y)


class LGMSFBridge(nn.Module):
    """Fuse P3 texture and P5 semantic features into an enhanced P5 feature."""

    def __init__(self, c3, c5, c_out, use_simam=True):
        super().__init__()
        self.texture = TextureBranch(c3, c_out)
        self.texture_down = nn.Sequential(
            LDSConv(c_out, c_out, k=3, s=2),
            LDSConv(c_out, c_out, k=3, s=2),
        )
        self.semantic = SemanticBranch(c5, c_out)
        self.fuse = FastNormFuse2()
        self.attn = SimAM() if use_simam else nn.Identity()

    def forward(self, xs):
        p3, p5 = xs
        texture = self.texture_down(self.texture(p3))
        semantic = self.semantic(p5)
        if texture.shape[-2:] != semantic.shape[-2:]:
            texture = F.interpolate(texture, size=semantic.shape[-2:], mode="nearest")
        return self.attn(self.fuse(texture, semantic))


# ---- LGMSF-Lite modules: end ----
