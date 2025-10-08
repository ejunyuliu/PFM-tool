from PFM import spang, util
import numpy as np
from scipy.special import hyp1f1
import logging
from scipy.ndimage import gaussian_filter

log = logging.getLogger('log')


def parallel_line(vox_dim=(130, 130, 130), px=(64, 64, 64), sigma=2, length=30, distance=6):
    phant = spang.Spang(np.zeros(px + (15,), dtype=np.float32), vox_dim=vox_dim)
    odf1 = bead_single([0, 1, 1], phant, kappa=30) + bead_single([0, -1, 1], phant,
                                                                 kappa=30)
    odf1 = odf1 / odf1[0]
    odf2 = bead_single([1, 0, 0], phant, kappa=5)
    odf2 = odf2 / odf2[0]

    center1 = (px[0] // 2, px[1] // 2 - distance // 2, px[2] // 2)
    center2 = (px[0] // 2, px[1] // 2 + distance // 2, px[2] // 2)

    mask1 = np.zeros(px)
    mask2 = np.zeros(px)

    mask1[(center1[0] - length // 2):(center1[0] + length // 2), center1[1], center1[2]] = 1
    mask2[(center2[0] - length // 2):(center2[0] + length // 2), center2[1], center2[2]] = 1

    mask1 = gaussian_filter(mask1, sigma=sigma)
    mask2 = gaussian_filter(mask2, sigma=sigma)

    phant.f = np.einsum('xyz,j->xyzj', mask1, odf1) + np.einsum('xyz,j->xyzj', mask2, odf2)

    phant.f = phant.f / phant.f[..., 0].max()

    return phant, center1, center2


def three_helix(vox_dim=(130, 130, 130), px=(64, 64, 64), s=20, radius=600, scale=[1, 1, 1]):
    phant = spang.Spang(np.zeros(px + (15,), dtype=np.float32), vox_dim=vox_dim)
    phant1 = helix_phantom(px=(s, s, s), radius=radius, pitch=1000,
                           vox_dim=vox_dim, max_l=4, center=(0, 0, 0),
                           normal=0, trange=(-2 * np.pi, 2 * np.pi))
    phant2 = helix_phantom(px=(s, s, s), radius=radius, pitch=1000,
                           vox_dim=vox_dim, max_l=4, center=(0, 0, 0),
                           normal=1, trange=(-2 * np.pi, 2 * np.pi))
    phant3 = helix_phantom(px=(s, s, s), radius=radius, pitch=1000,
                           vox_dim=vox_dim, max_l=4, center=(0, 0, 0),
                           normal=2, trange=(-2 * np.pi, 2 * np.pi))

    phant.f[0:s, 0:s, 0:s] = phant1.f * scale[0]
    phant.f[s:2 * s, s:2 * s, s:2 * s] = phant2.f * scale[1]
    phant.f[2 * s:3 * s, 2 * s:3 * s, 2 * s:3 * s] = phant3.f * scale[2]
    return phant


def double_helix(vox_dim=(130, 130, 130), px=(64, 64, 64), radius=600, rat=4, sigma=1):  # 双螺旋结构，两条平行反向的螺旋线
    # 1/rat表示螺线相差占螺旋线螺距的比例
    unit = px[2] // (2 * rat + 1)
    s = unit * rat * 2
    pitch = vox_dim[2] * unit * rat

    space = unit * vox_dim[2] / np.sqrt((4 * np.pi * radius) ** 2 + (s * vox_dim[2]) ** 2) * (4 * np.pi * radius)
    print(space)

    phant = spang.Spang(np.zeros(px + (15,), dtype=np.float32), vox_dim=vox_dim)
    phant1 = helix_phantom(px=(s, s, s), radius=radius, pitch=pitch,
                           vox_dim=vox_dim, max_l=4, center=(0, 0, 0),
                           normal=0, trange=(-2 * np.pi, 2 / rat * (rat - 1) * np.pi))

    phant2 = helix_phantom(px=(s, s, s), radius=radius, pitch=pitch,
                           vox_dim=vox_dim, max_l=4, center=(0, 0, 0),
                           normal=0, trange=(-2 / rat * (rat - 1) * np.pi, 2 * np.pi))

    phant.f[0:s, 0:s, unit:s + unit] += phant1.f
    phant.f[0:s, 0:s, 0:s] += phant2.f

    for j in range(15):
        phant.f[..., j] = gaussian_filter(phant.f[..., j], sigma=sigma)
    return phant, space


def helix_phantom(px=(20, 20, 20), vox_dim=(100, 100, 100), max_l=6,
                  trange=(-4 * np.pi, 4 * np.pi), nt=100, radius=700, pitch=1000,
                  cyl_rad=250, center=(0, 0, 0), normal=0, krange=(0, 5),
                  dtype=np.float32):
    log.info('Generating helix')
    t = np.linspace(trange[0], trange[1], nt)
    c = np.array([radius * np.cos(t), radius * np.sin(t), pitch * t / (2 * np.pi)]).T
    d = np.array([-radius * np.sin(t), radius * np.cos(t), pitch / (2 * np.pi) + 0 * t]).T
    d = d / np.linalg.norm(d, axis=-1)[..., None]  # normalize
    c = np.roll(c, normal, axis=-1) + center  # orient and recenter
    d = np.roll(d, normal, axis=-1)  # orient
    k = np.linspace(krange[0], krange[1], nt)  # watson kappa parameter
    return curve_phantom(c, d, k, cyl_rad=cyl_rad, vox_dim=vox_dim, px=px, max_l=max_l, dtype=dtype)


def curve_phantom(curve, direction, kappa,
                  px=(20, 20, 20), vox_dim=(100, 100, 100), cyl_rad=0.2, max_l=6,
                  dtype=np.float32):
    # Setup grid
    xyz = np.array(np.meshgrid(
        np.linspace(-(px[0] / 2) * vox_dim[0], (px[0] / 2) * vox_dim[0], px[0]),
        np.linspace(-(px[1] / 2) * vox_dim[1], (px[1] / 2) * vox_dim[1], px[1]),
        np.linspace(-(px[2] / 2) * vox_dim[2], (px[2] / 2) * vox_dim[2], px[2])))
    xyz = np.moveaxis(xyz, [0, 1], [-1, 1])

    # Calculate directions and kappas
    diff = xyz[:, :, :, None, :] - curve
    dist = np.linalg.norm(diff, axis=-1)
    min_dist = np.min(dist, axis=-1)  # min dist between grid points and curve
    t_index = np.argmin(dist, axis=-1)
    min_dir = direction[t_index]  # directions for closest point on curve
    min_k = kappa[t_index]  # kappas

    # Calculate watson 
    spang_shape = xyz.shape[0:-1] + (util.maxl2maxj(max_l),)
    spang1 = spang.Spang(np.zeros(spang_shape), vox_dim=vox_dim)
    dot = np.einsum('ijkl,ml->ijkm', min_dir, spang1.sphere.vertices)
    k = min_k[..., None]
    watson = np.exp(k * dot ** 2) / (4 * np.pi * hyp1f1(0.5, 1.5, k))
    watson_sh = np.einsum('ijkl,lm', watson, spang1.B)
    watson_sh = watson_sh / watson_sh[..., None, 0]  # Normalize

    # Cylinder mask
    mask = min_dist < cyl_rad
    spang1.f = np.einsum('ijkl,ijk->ijkl', watson_sh, mask).astype(dtype)

    return spang1


def guv(px=(32, 32, 32), vox_dim=(100, 100, 100), radius=12, sigma=2, bias=(0, 0, 0)):
    spang1 = spang.Spang(np.zeros(px + (15,), dtype=np.float32), vox_dim=vox_dim)

    X = (np.round(spang1.sphere.x * radius)).astype(int)
    Y = (np.round(spang1.sphere.y * radius)).astype(int)
    Z = (np.round(spang1.sphere.z * radius)).astype(int)

    for i in range(spang1.N):
        bead = bead_single(spang1.sphere.vertices[i], spang1, kappa=30)
        cx = px[0] // 2 + bias[0] + X[i]
        cy = px[1] // 2 + bias[1] + Y[i]
        cz = px[2] // 2 + bias[2] + Z[i]

        if (cx < 0) | (cx >= px[0]) | (cy < 0) | (cy >= px[1]) | (cz < 0) | (cz >= px[2]):
            continue

        spang1.f[cx, cy, cz, :] = bead

    if sigma is not None:
        for j in range(15):
            spang1.f[..., j] = gaussian_filter(spang1.f[..., j], sigma=sigma)

    return spang1


def guvs(px=(32, 32, 32), vox_dim=(100, 100, 100), num=10, sigma=1):
    spang1 = spang.Spang(np.zeros(px + (15,), dtype=np.float32), vox_dim=vox_dim)

    # np.random.seed(100)
    Centers = np.random.randint(5, px[0] - 5, size=(num, 3))

    # np.random.seed(10000)
    Radius = np.random.randint(2, 6, size=num)

    # np.random.seed(100000)
    Density = np.random.random(size=num)
    Density = Density / 2 + 0.5

    for i in range(num):
        radius = Radius[i]
        center = Centers[i, :]
        den = Density[i]

        X = (np.round(spang1.sphere.x * radius)).astype(int)
        Y = (np.round(spang1.sphere.y * radius)).astype(int)
        Z = (np.round(spang1.sphere.z * radius)).astype(int)

        for i in range(spang1.N):
            bead = bead_single(spang1.sphere.vertices[i], spang1, kappa=30)
            bead = bead / bead[0] * den
            spang1.f[center[0] + X[i], center[1] + Y[i], center[2] + Z[i], :] = bead

    for j in range(15):
        spang1.f[..., j] = gaussian_filter(spang1.f[..., j], sigma=sigma)

    return spang1


def bead_single(orientation, spang, kappa=None):
    # orientation sets the axis of rotational symmetry
    # kappa = None: all along single axis
    # kappa = 0: angularly uniform
    # kappa > 0: dumbbell
    # kappa < 0: pancake
    dot = np.dot(orientation, spang.sphere.vertices.T)
    watson = np.exp(kappa * dot ** 2) / (4 * np.pi * hyp1f1(0.5, 1.5, kappa))
    watson_sh = np.matmul(spang.B.T, watson)
    return watson_sh
