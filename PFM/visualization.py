import imageio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib import rc
# rc('text', usetex=True)
import numpy as np
# import vtk
import vtkmodules.all as vtk
import os
from dipy.viz import window, actor
from fury.colormap import colormap_lookup_table, create_colormap
from dipy.utils.optpkg import optional_package

from PFM import util

numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')


class ScaleMap:
    def __init__(self, min=0, max=1):
        self.min = min
        self.max = max
        self.window = max - min
        self.level = (max - min) / 2

        if self.min == self.max:
            print("Warning: min and max are equal. Setting min=0, max=1.")
            self.min = 0
            self.max = 1
            self.window = 1
            self.level = 0.5

    def mapper(self, x):
        out = np.zeros_like(x)
        out = (x - self.min) / self.window
        out[x < self.min] = 0
        out[x > self.max] = 1
        return out


def plot_parallels(raw_data, out_path='out/', outer_box=True, axes=True,
                   clip_neg=False, size=(600, 600), mask=None, scale=1,
                   azimuth=0, elevation=0, zoom=1.7):
    # Prepare output
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Mask
    if mask is None:
        mask = np.ones(raw_data.shape)
    raw_data = raw_data * mask

    # Render
    ren = window.Scene()
    ren.background([1, 1, 1])

    # Add visuals to renderer
    data = np.zeros(raw_data.shape)

    # X MIP
    data[data.shape[0] // 2, :, :] = np.max(raw_data, axis=0)
    slice_actorx = actor.slicer(data, value_range=(0, 1), interpolation='nearest')
    slice_actorx.display(slice_actorx.shape[0] // 2, None, None)
    ren.add(slice_actorx)

    # Y MIP
    data[:, data.shape[1] // 2, :] = np.max(raw_data, axis=1)
    slice_actory = actor.slicer(data, value_range=(0, 1), interpolation='nearest')
    slice_actory.display(None, slice_actory.shape[1] // 2, None)
    ren.add(slice_actory)

    # Z MIP
    data[:, :, data.shape[2] // 2] = np.max(raw_data, axis=-1)
    slice_actorz = actor.slicer(data, value_range=(0, 1), interpolation='nearest')
    slice_actorz.display(None, None, slice_actorz.shape[2] // 2)
    ren.add(slice_actorz)

    X = raw_data.shape[0] - 1
    Y = raw_data.shape[1] - 1
    Z = raw_data.shape[2] - 1

    if outer_box:
        c = np.array([0, 0, 0])
        ren.add(actor.line([np.array([[0, 0, 0], [X, 0, 0], [X, Y, 0], [0, Y, 0],
                                      [0, 0, 0], [0, Y, 0], [0, Y, Z], [0, 0, Z],
                                      [0, 0, 0], [X, 0, 0], [X, 0, Z], [0, 0, Z]])], colors=c))
        ren.add(actor.line([np.array([[X, 0, Z], [X, Y, Z], [X, Y, 0], [X, Y, Z],
                                      [0, Y, Z]])], colors=c))
    NN = np.max([X, Y, Z])
    # Add invisible actors to set FOV
    ren.add(actor.line([np.array([[0, 0, 0], [NN, 0, 0]])], colors=np.array([1, 1, 1]), linewidth=1))
    ren.add(actor.line([np.array([[0, 0, 0], [0, NN, 0]])], colors=np.array([1, 1, 1]), linewidth=1))
    ren.add(actor.line([np.array([[0, 0, 0], [0, 0, NN]])], colors=np.array([1, 1, 1]), linewidth=1))
    # Add colored axes
    if axes:
        ren.add(actor.line([np.array([[0, 0, 0], [NN / 10, 0, 0]])], colors=np.array([1, 0, 0]), linewidth=4))
        ren.add(actor.line([np.array([[0, 0, 0], [0, NN / 10, 0]])], colors=np.array([0, 1, 0]), linewidth=4))
        ren.add(actor.line([np.array([[0, 0, 0], [0, 0, NN / 10]])], colors=np.array([0, 0, 1]), linewidth=4))

    # Setup vtk renderers
    renWin = vtk.vtkRenderWindow()
    renWin.SetOffScreenRendering(1)
    renWin.AddRenderer(ren)
    renWin.SetSize(size[0], size[1])
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    ren.ResetCamera()
    ren.azimuth(azimuth)
    ren.elevation(elevation)

    # writer = vtk.vtkPNGWriter()
    writer = vtk.vtkTIFFWriter()
    writer.SetCompressionToNoCompression()

    az = 0

    filenames = ['yz', 'xy', 'xz']
    zooms = [zoom, 1.0, 1.0]
    azs = [90, -90, 0]
    els = [0, 0, 90]
    for i in range(3):
        ren.projection(proj_type='parallel')
        ren.zoom(zooms[i])
        ren.azimuth(azs[i])
        ren.elevation(els[i])
        ren.reset_clipping_range()
        renderLarge = vtk.vtkRenderLargeImage()
        renderLarge.SetMagnification(1)
        renderLarge.SetInput(ren)
        renderLarge.Update()
        writer.SetInputConnection(renderLarge.GetOutputPort())
        writer.SetFileName(out_path + filenames[i] + '.tif')
        writer.Write()


def plot5d(filename, data, row_labels=None, col_labels=None, yscale_label=None,
           force_bwr=False, normalize=False):
    if np.min(data) < 0 or force_bwr:
        colormap = 'bwr'
        vmin = -1
        vmax = 1
    else:
        colormap = 'gray'
        vmin = 0
        vmax = 1

    if normalize:
        data = data / np.max(data)

    inches = 4
    rows = data.shape[-1]
    cols = data.shape[-2] + 1
    widths = [1] * cols
    heights = [1] * rows
    M = np.max(data.shape)
    x_frac = data.shape[0] / M
    f = plt.figure(figsize=(inches * np.sum(widths), inches * np.sum(heights)))
    spec = gridspec.GridSpec(ncols=cols, nrows=rows, width_ratios=widths,
                             height_ratios=heights, hspace=0.075, wspace=0.075)
    for row in range(rows):
        print('Plotting 5D row ' + str(row))
        for col in range(cols):
            if col != cols - 1:
                plot_parallels(data[:, :, :, col, row], out_path='parallels/', outer_box=False,
                               axes=False, clip_neg=False, azimuth=0,
                               elevation=0)
                plot_images(['parallels/yz.tif', 'parallels/xy.tif', 'parallels/xz.tif'],
                            f, spec, row, col,
                            col_labels=col_labels, row_labels=row_labels,
                            vmin=vmin, vmax=vmax, colormap=colormap,
                            rows=rows, cols=cols, x_frac=x_frac, yscale_label=yscale_label)

    f.savefig(filename, bbox_inches='tight')


def plot_images(images, f, spec, row, col, col_labels, row_labels, vmin, vmax,
                colormap, rows, cols, x_frac, yscale_label, pos=(-0.05, 1.05, 0.5, 0.5),
                bar=True, bar_label=''):
    mini_spec = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=spec[row, col], hspace=0.1, wspace=0.1)
    for a in range(2):
        for b in range(2):
            ax = f.add_subplot(mini_spec[a, b])
            if a == 0 and b == 0:
                image = images[0]
            if a == 0 and b == 1:
                image = images[1]
            if a == 1 and b == 1:
                image = images[2]
                draw_annotations(ax, row, col, row_labels, col_labels, pos=pos)
            if a == 1 and b == 0:
                image = None
                if bar:
                    plot_colorbar(ax, spec, vmin, vmax, colormap, bar_label=bar_label)
                else:
                    im = imageio.imread(os.path.join(os.path.dirname(__file__), '../assets/rgb.png'))
                    bb = ax.get_position()
                    bb.y0 += 0.04
                    bb.y1 += 0.04
                    ax.set_position(bb)
                    ax.imshow(im, interpolation='none', origin='upper', extent=[-1, 1, -1, 1], aspect=1)
                    s = 3
                    ax.set_xlim([-s, s])
                    ax.set_ylim([-s, s])
                    ax.axis('off')
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    r = 1.2
                    zs = 0.2
                    ax.annotate('$x$', xy=(0, 0), xytext=(-r * np.sqrt(3) / 2, -r * np.sqrt(3) / 2 + zs),
                                xycoords='data', textcoords='data', va='center', ha='center', fontsize=12)
                    ax.annotate('$y$', xy=(0, 0), xytext=(r * np.sqrt(3) / 2, -r * np.sqrt(3) / 2 + zs),
                                xycoords='data', textcoords='data', va='center', ha='center', fontsize=12)
                    ax.annotate('$z$', xy=(0, 0), xytext=(0, r + zs), xycoords='data', textcoords='data', va='center',
                                ha='center', fontsize=12)
                    ax.annotate(bar_label, xy=(0, 0), xytext=(0, -1.5), xycoords='data', textcoords='data', va='top',
                                ha='center', fontsize=12)
            if image is not None:
                im = imageio.imread(image)
                ax.imshow(im, interpolation='none', origin='upper', extent=[-24, 24, -24, 24], aspect=1)
                ax.axis('off')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            if col == (cols - 1) and row == rows - 1 and a == 1 and b == 1 and yscale_label is not None:
                ax.annotate(yscale_label, xy=(x_frac / 2, -0.2), xytext=(x_frac / 2, -0.2), xycoords='axes fraction',
                            textcoords='axes fraction', va='center', ha='center', fontsize=12)
                ax.annotate('', xy=(0, -0.1), xytext=(x_frac, -0.1), xycoords='axes fraction',
                            textcoords='axes fraction', va='center',
                            arrowprops=dict(arrowstyle='|-|, widthA=0.2, widthB=0.2', shrinkA=0.05, shrinkB=0.05,
                                            lw=0.5))


def plot_colorbar(ax, spec, vmin, vmax, colormap, bar_label=''):
    X, Y = np.meshgrid(np.linspace(vmin, vmax, 100),
                       np.linspace(vmin, vmax, 100))
    bb = ax.get_position()
    bb.x0 += 0.02
    bb.x1 += -0.02
    bb.y0 += 0.02
    bb.y1 += 0.02
    ax.set_position(bb)
    ax.imshow(X, cmap=colormap, vmin=vmin, vmax=vmax, interpolation='none',
              extent=[vmin, vmax, vmin, vmax], origin='lower', aspect=1 / 12)
    ax.set_xlim([vmin, vmax])
    ax.set_ylim([vmin, vmax])
    ax.tick_params(direction='out', left=True, right=False, labelsize=12, width=0.5)
    ax.yaxis.set_ticks([])
    ax.xaxis.tick_top()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    if vmin == -1:
        ax.xaxis.set_ticks([-1.0, 0, 1.0])
    else:
        ax.xaxis.set_ticks([0, 1])
    ax.annotate(bar_label, xy=(0, 0), xytext=(0.5, -2), xycoords='data', textcoords='data', va='top', ha='center',
                fontsize=12)


def draw_annotations(ax, row, col, row_labels, col_labels, pos=(-0.05, 1.05, 0.5, 0.5)):
    xc = pos[0]
    yc = pos[1]
    d1 = pos[2]
    d2 = pos[3]
    o = 0.05
    ax.annotate('', xy=(xc, yc), xytext=(xc + d1, yc), xycoords='axes fraction', textcoords='axes fraction',
                va='center', arrowprops=dict(arrowstyle="<-", shrinkB=0, lw=0.5))
    ax.annotate('', xy=(xc, yc), xytext=(xc - d1, yc), xycoords='axes fraction', textcoords='axes fraction',
                ha='center', arrowprops=dict(arrowstyle="<-", shrinkB=0, lw=0.5))
    ax.annotate('', xy=(xc, yc), xytext=(xc, yc + d1), xycoords='axes fraction', textcoords='axes fraction',
                ha='center', arrowprops=dict(arrowstyle="<-", shrinkB=0, lw=0.5))
    ax.annotate('', xy=(xc, yc), xytext=(xc, yc - d1), xycoords='axes fraction', textcoords='axes fraction',
                va='center', arrowprops=dict(arrowstyle="<-", shrinkB=0, lw=0.5))
    ax.annotate('$x$', xy=(xc, yc), xytext=(xc + d2 + o, yc), xycoords='axes fraction', textcoords='axes fraction',
                va='center', ha='center', fontsize=12)
    ax.annotate('$z$', xy=(xc, yc), xytext=(xc - d2 - o, yc), xycoords='axes fraction', textcoords='axes fraction',
                va='center', ha='center', fontsize=12)
    ax.annotate('$y$', xy=(xc, yc), xytext=(xc, yc + d2 + o), xycoords='axes fraction', textcoords='axes fraction',
                va='center', ha='center', fontsize=12)
    ax.annotate('$z$', xy=(xc, yc), xytext=(xc, yc - d2 - o), xycoords='axes fraction', textcoords='axes fraction',
                va='center', ha='center', fontsize=12)
    if col_labels is not None:
        ax.annotate(col_labels[row, col], xy=(xc, yc), xytext=(0, 2.2), xycoords='axes fraction',
                    textcoords='axes fraction', va='center', ha='center', fontsize=12)
    if col == 0 and row_labels is not None:
        ax.annotate(row_labels[row], xy=(xc, yc), xytext=(-1.35, 1), xycoords='axes fraction',
                    textcoords='axes fraction', va='center', ha='center', fontsize=12, rotation=90)


def odf_sparse(odfsh, Binv, affine=None, mask=None, sphere=None, scale=2.2,
               norm=True, radial_scale=True, opacity=1., colormap='plasma',
               global_cm=False, scalemap=ScaleMap(), odf_sphere=False,
               flat=False, normalize=True):
    if mask is None:
        mask = np.ones(odfsh.shape[:3], dtype=np.bool)
    else:
        mask = mask.astype(np.bool)

    szx, szy, szz = odfsh.shape[:3]

    class OdfSlicerActor(vtk.vtkLODActor):

        def display_extent(self, x1, x2, y1, y2, z1, z2):
            tmp_mask = np.zeros(odfsh.shape[:3], dtype=np.bool)
            tmp_mask[x1:x2 + 1, y1:y2 + 1, z1:z2 + 1] = True
            tmp_mask = np.bitwise_and(tmp_mask, mask)

            self.mapper = _odf_slicer_mapper(odfsh=odfsh, Binv=Binv,
                                             affine=affine,
                                             mask=tmp_mask,
                                             sphere=sphere,
                                             scale=scale,
                                             norm=norm,
                                             radial_scale=radial_scale,
                                             opacity=opacity,
                                             colormap=colormap,
                                             global_cm=global_cm,
                                             scalemap=scalemap,
                                             odf_sphere=odf_sphere,
                                             normalize=normalize)
            self.SetMapper(self.mapper)

        def display(self, x=None, y=None, z=None):
            if x is None and y is None and z is None:
                self.display_extent(0, szx - 1, 0, szy - 1, 0, szz - 1)
            if x is not None:
                self.display_extent(x, x, 0, szy - 1, 0, szz - 1)
            if y is not None:
                self.display_extent(0, szx - 1, y, y, 0, szz - 1)
            if z is not None:
                self.display_extent(0, szx - 1, 0, szy - 1, z, z)

    odf_actor = OdfSlicerActor()
    odf_actor.display_extent(0, szx - 1, 0, szy - 1, 0, szz - 1)

    if flat:
        odf_actor.GetProperty().SetAmbient(1)
        odf_actor.GetProperty().SetDiffuse(0)

    return odf_actor


def _odf_slicer_mapper(odfsh, Binv, affine=None, mask=None, sphere=None,
                       scale=2.2, norm=True, radial_scale=True, opacity=1.,
                       colormap='plasma', global_cm=False,
                       scalemap=ScaleMap(), odf_sphere=False, normalize=True):
    if mask is None:
        mask = np.ones(odfsh.shape[:3])

    ijk = np.ascontiguousarray(np.array(np.nonzero(mask)).T)

    if len(ijk) == 0:
        return None

    if affine is not None:
        ijk = np.ascontiguousarray(apply_affine(affine, ijk))

    faces = np.asarray(sphere.faces, dtype=int)
    vertices = sphere.vertices

    all_xyz = []
    all_faces = []
    all_ms = []
    masked_sh = odfsh[ijk[:, 0], ijk[:, 1], ijk[:, 2]]  # Assemble masked sh
    # masked_sh = np.einsum('nj,n->nj', masked_sh, 1 / masked_sh[:, 0])
    masked_sh_scaled = np.einsum('ij,i->ij', masked_sh,
                                 scalemap.mapper(masked_sh[:, 0]) / masked_sh[:, 0])  # Scale mapping
    radii = np.einsum('vj,pj->vp', Binv.T, masked_sh_scaled)  # Radii
    if odf_sphere:
        masked_radii = np.einsum('vj,pj->vp', np.ones_like(Binv.T), masked_sh_scaled)  # Radii
    else:
        masked_radii = radii
    if normalize:
        xyz_vertices = np.einsum('ij,ik->ikj', vertices, masked_radii) * scale / np.max(masked_radii) + ijk  # Vertices
    else:
        xyz_vertices = np.einsum('ij,ik->ikj', vertices, masked_radii) * scale + ijk  # Vertices

    all_xyz = xyz_vertices.reshape(-1, xyz_vertices.shape[-1], order='F')  # Reshape
    all_xyz_vtk = numpy_support.numpy_to_vtk(all_xyz, deep=True)  # Convert to vtk

    # Assemble faces
    for (k, center) in enumerate(ijk):
        all_faces.append(faces + k * xyz_vertices.shape[0])

    all_faces = np.concatenate(all_faces)
    all_faces = np.hstack((3 * np.ones((len(all_faces), 1)),
                           all_faces))
    ncells = len(all_faces)
    all_faces = np.ascontiguousarray(all_faces.ravel(), dtype='i8')
    all_faces_vtk = numpy_support.numpy_to_vtkIdTypeArray(all_faces, deep=True)

    all_ms = radii
    if global_cm:
        all_ms[0, 0] = -np.max(radii)  # For setting colors
    all_ms = all_ms.flatten(order='F')

    points = vtk.vtkPoints()
    points.SetData(all_xyz_vtk)

    cells = vtk.vtkCellArray()
    cells.SetCells(ncells, all_faces_vtk)

    if colormap is not None:
        cols = create_colormap(all_ms.ravel(), colormap)
        if global_cm:
            cols[0] = cols[1]

        vtk_colors = numpy_support.numpy_to_vtk(
            np.asarray(255 * cols),
            deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR)

        vtk_colors.SetName("Colors")

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)

    if colormap is not None:
        polydata.GetPointData().SetScalars(vtk_colors)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    return mapper


def peak_slicer_sparse(odfsh, Binv, vertices, mask=None, affine=None, scale=1, peak_scale=1,
                       colors=None, opacity=1., linewidth=0.1, lod=False, balls=False,
                       lod_points=10 ** 4, lod_points_size=30,
                       scalemap=ScaleMap(), normalize=True):
    xyz = np.ascontiguousarray(np.array(np.nonzero(mask)).T)
    masked_sh = odfsh[mask]  # Assemble masked sh
    masked_sh_scaled = np.einsum('ij,i->ij', masked_sh,
                                 scalemap.mapper(masked_sh[:, 0]) / masked_sh[:, 0])  # Scale mapping
    masked_radii = np.einsum('vj,pj->vp', Binv.T, masked_sh_scaled)  # Radii
    index = np.argmax(masked_radii, axis=0)
    peak_dirs = vertices[index]
    peak_values = np.amax(masked_radii, axis=0)
    if normalize:
        peak_values = peak_values * scale / np.max(peak_values) * peak_scale
        # peak_values = peak_values / peak_values * scale * peak_scale
    else:
        peak_values = peak_values * scale * peak_scale
    list_dirs = []
    for (i, peak) in enumerate(peak_values):
        if peak > 0:
            symm = np.vstack((-peak_dirs[i, :] * peak + xyz[i],
                              peak_dirs[i, :] * peak + xyz[i]))
            list_dirs.append(symm)

    # CUSTOM COLORS
    def orient2rgb(v):
        M = np.array([[1 / np.sqrt(2), 0, 1 / np.sqrt(2)], [0, 1, 0], [-1 / np.sqrt(2), 0, 1 / np.sqrt(2)]])
        vv = np.dot(M, v)
        return np.abs(vv / np.linalg.norm(vv))

    col_list = [orient2rgb(streamline[-1] - streamline[0]) for streamline in list_dirs]
    cols_arr = np.vstack(col_list)

    if balls == False:
        return actor.streamtube(list_dirs, colors=cols_arr,
                                opacity=opacity, linewidth=linewidth * scale,  # multiply linewidth for narrower lines
                                lod=lod, lod_points=lod_points,
                                lod_points_size=lod_points_size)
    else:
        return actor.point(xyz, cols_arr, point_radius=0.2)


def peak_mp_slicer_sparse(odfsh, Binv, vertices, mask=None, affine=None, scale=1, peak_scale=1,
                          colors=None, opacity=1., linewidth=0.1, lod=False,
                          lod_points=10 ** 4, lod_points_size=30, balls=False,
                          scalemap=ScaleMap(), normalize=True):
    from dipy.data import get_sphere
    sphere = get_sphere('symmetric724')
    sph_len = len(sphere.theta)
    B = np.zeros((sph_len, 15))
    for (n, j), x in np.ndenumerate(B):
        l, m = util.j2lm(j)
        B[n, j] = util.spZnm(l, m, sphere.theta[n], sphere.phi[n])
    Binv = np.linalg.pinv(B, rcond=1e-15)
    BinvT = Binv.T

    def compute_MP(Binv_MP, BinvT, sphere, n):
        vec = sphere.vertices[n, :]
        MP = np.einsum('vp,p->v', sphere.vertices, vec)
        MP = np.abs(MP)
        Binv_MP[n, :] = np.einsum('vj,v->j', BinvT, MP)

    BinvT_MP = np.zeros_like(BinvT)
    from joblib import Parallel, delayed
    Parallel(n_jobs=-1, backend='threading')([delayed(compute_MP)(BinvT_MP, BinvT, sphere, n) for n in range(sph_len)])

    xyz = np.ascontiguousarray(np.array(np.nonzero(mask)).T)
    masked_sh = odfsh[mask]  # Assemble masked sh
    masked_radii = np.einsum('vj,pj->vp', BinvT_MP, masked_sh)  # Radii
    index = np.argmax(masked_radii, axis=0)
    peak_dirs = sphere.vertices[index]
    peak_values = np.amax(masked_radii, axis=0)

    if normalize:
        peak_values = peak_values * scale / np.max(peak_values) * peak_scale
    else:
        peak_values = peak_values * scale * peak_scale

    list_dirs = []
    for (i, peak) in enumerate(peak_values):
        if peak > 0:
            symm = np.vstack((-peak_dirs[i, :] * peak + xyz[i],
                              peak_dirs[i, :] * peak + xyz[i]))
            list_dirs.append(symm)

    # CUSTOM COLORS
    def orient2rgb(v):
        # M = np.array([[1 / np.sqrt(2), 0, 1 / np.sqrt(2)], [0, 1, 0], [-1 / np.sqrt(2), 0, 1 / np.sqrt(2)]])
        # vv = np.dot(M, v)
        vv = v
        return np.abs(vv / np.linalg.norm(vv))

    col_list = [orient2rgb(streamline[-1] - streamline[0]) for streamline in list_dirs]
    cols_arr = np.vstack(col_list)

    if balls == False:
        return actor.streamtube(list_dirs, colors=cols_arr,
                                opacity=opacity, linewidth=linewidth * scale,  # multiply linewidth for narrower lines
                                lod=lod, lod_points=lod_points,
                                lod_points_size=lod_points_size)
    else:
        return actor.point(xyz, cols_arr, point_radius=1)


def principal_slicer_sparse(odfsh, Binv, vertices, mask=None, affine=None, scale=1,
                            colors=None, opacity=1., linewidth=0.1,
                            lod=False, lod_points=10 ** 4, lod_points_size=30):
    xyz = np.ascontiguousarray(np.array(np.nonzero(mask)).T)

    # Mask
    masked_sh = odfsh[mask]  # Assemble masked sh

    # Calculate evals, evecs, principal
    M = np.load(os.path.join(os.path.dirname(__file__), 'harmonics/sh2tensor.npy'))
    Di = np.einsum('il,ml->im', masked_sh[:, 0:6], M)
    D = np.zeros((Di.shape[0],) + (3, 3), dtype=np.float32)
    D[..., 0, 0] = Di[..., 0];
    D[..., 0, 1] = Di[..., 3];
    D[..., 0, 2] = Di[..., 5];
    D[..., 1, 0] = Di[..., 3];
    D[..., 1, 1] = Di[..., 1];
    D[..., 1, 2] = Di[..., 4];
    D[..., 2, 0] = Di[..., 5];
    D[..., 2, 1] = Di[..., 4];
    D[..., 2, 2] = Di[..., 2];
    evals, evecs = np.linalg.eigh(D)
    peak_dirs = evecs[..., -1]
    peak_values = evals[..., -1] * scale / np.max(evals)
    list_dirs = []
    for (i, peak) in enumerate(peak_values):
        if peak > 0:
            symm = np.vstack((-peak_dirs[i, :] * peak + xyz[i],
                              peak_dirs[i, :] * peak + xyz[i]))
            list_dirs.append(symm)
    return actor.streamtube(list_dirs, colors=colors,
                            opacity=opacity, linewidth=linewidth * scale,
                            lod=lod, lod_points=lod_points,
                            lod_points_size=lod_points_size)


def tensor_slicer_sparse(odfsh, affine=None, mask=None, sphere=None, scale=2.2,
                         norm=True, opacity=1., scalar_colors=None):
    if mask is None:
        mask = np.ones(odfsh.shape[:3], dtype=np.bool)
    else:
        mask = mask.astype(np.bool)

    szx, szy, szz = odfsh.shape[:3]

    class TensorSlicerActor(vtk.vtkLODActor):

        def display_extent(self, x1, x2, y1, y2, z1, z2):
            tmp_mask = np.zeros(odfsh.shape[:3], dtype=np.bool)
            tmp_mask[x1:x2 + 1, y1:y2 + 1, z1:z2 + 1] = True
            tmp_mask = np.bitwise_and(tmp_mask, mask)

            self.mapper = _tensor_slicer_mapper(odfsh,
                                                affine=affine,
                                                mask=tmp_mask,
                                                sphere=sphere,
                                                scale=scale,
                                                norm=False,
                                                opacity=opacity,
                                                scalar_colors=scalar_colors)
            self.SetMapper(self.mapper)

        def display(self, x=None, y=None, z=None):
            if x is None and y is None and z is None:
                self.display_extent(0, szx - 1, 0, szy - 1, 0, szz - 1)
            if x is not None:
                self.display_extent(x, x, 0, szy - 1, 0, szz - 1)
            if y is not None:
                self.display_extent(0, szx - 1, y, y, 0, szz - 1)
            if z is not None:
                self.display_extent(0, szx - 1, 0, szy - 1, z, z)

    tensor_actor = TensorSlicerActor()
    tensor_actor.display_extent(0, szx - 1, 0, szy - 1, 0, szz - 1)
    return tensor_actor


def _tensor_slicer_mapper(odfsh, affine=None, mask=None, sphere=None, scale=2.2,
                          norm=False, opacity=1., scalar_colors=None):
    if mask is None:
        mask = np.ones(odfsh.shape[:3])

    ijk = np.ascontiguousarray(np.array(np.nonzero(mask)).T)
    if len(ijk) == 0:
        return None

    if affine is not None:
        ijk = np.ascontiguousarray(apply_affine(affine, ijk))

    faces = np.asarray(sphere.faces, dtype=int)
    vertices = sphere.vertices

    # Mask
    masked_sh = odfsh[ijk[:, 0], ijk[:, 1], ijk[:, 2]]  # Assemble masked sh

    # Normalize
    if norm:
        masked_sh = masked_sh / np.max(masked_sh[:, 0])

    # Calculate evals, evecs, principal
    M = np.load(os.path.join(os.path.dirname(__file__), 'harmonics/sh2tensor.npy'))
    Di = np.einsum('il,ml->im', masked_sh[:, 0:6], M)
    D = np.zeros((Di.shape[0],) + (3, 3), dtype=np.float32)
    D[..., 0, 0] = Di[..., 0];
    D[..., 0, 1] = Di[..., 3];
    D[..., 0, 2] = Di[..., 5];
    D[..., 1, 0] = Di[..., 3];
    D[..., 1, 1] = Di[..., 1];
    D[..., 1, 2] = Di[..., 4];
    D[..., 2, 0] = Di[..., 5];
    D[..., 2, 1] = Di[..., 4];
    D[..., 2, 2] = Di[..., 2];
    evals2, evecs2 = np.linalg.eigh(D)
    pr2 = evecs2[..., -1]

    # Calculate vertices
    masked_radii = np.einsum('ij,kj->kij', evals2, vertices)  # Scale
    masked_radii2 = np.einsum('ijk,lik->lij', evecs2, masked_radii)  # Rotate
    xyz_vertices = masked_radii2 * scale + ijk  # Vertices

    all_xyz2 = xyz_vertices.reshape(-1, xyz_vertices.shape[-1], order='F')  # Reshape
    all_xyz_vtk = numpy_support.numpy_to_vtk(np.ascontiguousarray(all_xyz2), deep=True)  # Convert to vtk

    from fury.colormap import orient2rgb
    scalar_colors2 = orient2rgb(pr2.reshape(-1, pr2.shape[-1])).reshape(pr2.shape)
    cols2 = np.zeros((scalar_colors2.shape[0],) + sphere.vertices.shape, dtype='f4')

    # Faces and colors
    all_faces = []
    for (k, center) in enumerate(ijk):
        all_faces.append(faces + k * xyz_vertices.shape[0])

    cols01 = np.einsum('ij,ikj->ikj', scalar_colors2, np.ones((scalar_colors2.shape[0],) + sphere.vertices.shape))
    cols = np.interp(cols01, [0, 1], [0, 255]).astype('ubyte')

    all_faces = np.concatenate(all_faces)
    all_faces = np.hstack((3 * np.ones((len(all_faces), 1)),
                           all_faces))
    ncells = len(all_faces)

    all_faces = np.ascontiguousarray(all_faces.ravel(), dtype='i8')
    all_faces_vtk = numpy_support.numpy_to_vtkIdTypeArray(all_faces,
                                                          deep=True)

    points = vtk.vtkPoints()
    points.SetData(all_xyz_vtk)

    cells = vtk.vtkCellArray()
    cells.SetCells(ncells, all_faces_vtk)

    cols = np.ascontiguousarray(
        np.reshape(cols, (cols.shape[0] * cols.shape[1],
                          cols.shape[2])), dtype='f4')

    vtk_colors = numpy_support.numpy_to_vtk(
        cols,
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR)

    vtk_colors.SetName("Colors")

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    polydata.GetPointData().SetScalars(vtk_colors)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    return mapper


def _makeNd(array, ndim):
    """Pads as many 1s at the beginning of array's shape as are need to give
    array ndim dimensions."""
    new_shape = (1,) * (ndim - array.ndim) + array.shape
    return array.reshape(new_shape)


def density_slicer(density, scalemap=ScaleMap(), ss=1):
    from scipy import ndimage
    density = ndimage.zoom(density, ss)

    # Set opacity
    vol = np.interp(np.swapaxes(density, 0, 2), [scalemap.min, scalemap.max], [0, 255])
    vol = vol.astype('uint8')

    X, Y, Z = density.shape

    dataImporter = vtk.vtkImageImport()
    data_string = vol.tostring()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)
    dataImporter.SetDataExtent(0, X - 1, 0, Y - 1, 0, Z - 1)
    dataImporter.SetWholeExtent(0, X - 1, 0, Y - 1, 0, Z - 1)
    dataImporter.SetDataSpacing(1 / ss, 1 / ss, 1 / ss)

    # Create transfer mapping scalar value to opacity
    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(0, 0)  # Previously 0.0
    opacityTransferFunction.AddPoint(255, 0.9)

    # Create transfer mapping scalar value to color
    colorTransferFunction = vtk.vtkColorTransferFunction()
    colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
    colorTransferFunction.AddRGBPoint(255.0, 1, 1, 1)

    # The property describes how the data will look
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorTransferFunction)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)
    # volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()

    # The mapper / ray cast function know how to render the data
    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetBlendModeToMaximumIntensity()
    volumeMapper.SetSampleDistance(0.01)
    # volumeMapper.SetAutoAdjustSampleDistances(0)
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

    # The class vtkVolume is used to pair the preaviusly declared volume as well as the properties to be used when rendering that volume.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    # The volume holds the mapper and the property and
    # can be used to position/orient the volume
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    return volume


# def draw_unlit_line(ren, coords, colors, lw=0.5, scale=1.0, streamtube=True):
#     if streamtube:
#         act = actor.streamtube(coords, colors=colors, linewidth=scale*lw, lod=False)
#     else:
#         act = actor.line(coords, colors=colors, linewidth=scale*lw)
#     act.GetProperty().SetLighting(0)
#     act.GetProperty().SetOpacity(1)
#     ren.AddActor(act)

def draw_unlit_line(ren, coords, colors, lw=0.5, scale=1.0):
    for i in range(len(coords[0]) - 1):
        # Create a line
        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(coords[0][i, :])
        lineSource.SetPoint2(coords[0][i + 1, :])

        # Setup actor and mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputConnection(lineSource.GetOutputPort())

        # Create tube filter
        tubeFilter = vtk.vtkTubeFilter()
        tubeFilter.SetInputConnection(lineSource.GetOutputPort())
        tubeFilter.SetRadius(lw)
        tubeFilter.SetNumberOfSides(50)
        tubeFilter.Update()

        # Setup actor and mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(tubeFilter.GetOutputPort())

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(lineSource.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(tubeMapper)
        # actor.GetProperty().SetLineWidth(100*lw)
        actor.GetProperty().SetColor(colors[i])
        actor.GetProperty().SetLighting(0)

        actor.GetProperty().SetLighting(0)
        actor.GetProperty().SetOpacity(1)
        ren.AddActor(actor)


def draw_outer_box(ren, roi, line_color, lw=0.05):  # 0.3
    X0, Y0, Z0 = roi[0]
    X1, Y1, Z1 = roi[1]
    Nmax = np.max([X1 - X0, Y1 - Y0, Z1 - Z0])
    scale = Nmax / 63.0
    lines = [[np.array([[X0, Y0, Z0], [X1, Y0, Z0]], dtype=np.float)],
             [np.array([[X0, Y0, Z0], [X0, Y1, Z0]], dtype=np.float)],
             [np.array([[X0, Y0, Z0], [X0, Y0, Z1]], dtype=np.float)],
             [np.array([[X1, Y1, Z1], [X1, Y1, Z0]], dtype=np.float)],
             [np.array([[X1, Y1, Z1], [X0, Y1, Z1]], dtype=np.float)],
             [np.array([[X1, Y1, Z1], [X1, Y0, Z1]], dtype=np.float)],
             [np.array([[X1, Y0, Z0], [X1, Y1, Z0]], dtype=np.float)],
             [np.array([[X1, Y0, Z0], [X1, Y0, Z1]], dtype=np.float)],
             [np.array([[X0, Y1, Z0], [X1, Y1, Z0]], dtype=np.float)],
             [np.array([[X0, Y1, Z0], [X0, Y1, Z1]], dtype=np.float)],
             [np.array([[X0, Y0, Z1], [X1, Y0, Z1]], dtype=np.float)],
             [np.array([[X0, Y0, Z1], [X0, Y1, Z1]], dtype=np.float)]]
    for line in lines:
        draw_unlit_line(ren, line, [line_color], lw=lw, scale=scale)


def draw_profile(ren, X, Y, Z, profile, color, lw=0.3):
    Nmax = np.max([X, Y, Z])
    scale = Nmax / 63.0
    for i in range(profile.shape[0] - 1):
        line = np.array([profile[i, :], profile[i + 1, :]], dtype=np.float)
        draw_unlit_line(ren, [line], [color], lw=lw, scale=scale)


def draw_scale_bar(ren, X, Y, Z, line_color):
    Nmax = np.max([X, Y, Z])
    scale = Nmax / 63.0
    draw_unlit_line(ren, [np.array([[X, 0, -Z // 40], [X, Y, -Z // 40]])], [line_color], lw=0.3, scale=scale)
    # draw_unlit_line(ren, [np.array([[X,Y,-Z//40 + Z//60],[X,Y,-Z//40 - Z//60]])], line_color, lw=0.3, scale=scale)
    # draw_unlit_line(ren, [np.array([[X,0,-Z//40 + Z//60],[X,0,-Z//40 - Z//60]])], line_color, lw=0.3, scale=scale)


def draw_axes(ren, roi, viz_type, lw_rat=1):
    lw = 0.75 * lw_rat

    X0, Y0, Z0 = roi[0]
    X1, Y1, Z1 = roi[1]
    Nmin = np.min([X1 - X0, Y1 - Y0, Z1 - Z0])
    Nmax = np.max([X1 - X0, Y1 - Y0, Z1 - Z0])
    scale = Nmax / 63.0

    if (viz_type == 'Density') | (viz_type == 'ODF'):
        return

    if viz_type == 'Peak_MP':
        R = np.array([[X0, Y0, Z0], [X0 + Nmax / 8, Y0, Z0]])
        G = np.array([[X0, Y0, Z0], [X0, Y0 + Nmax / 8, Z0]])
        B = np.array([[X0, Y0, Z0], [X0, Y0, Z0 + Nmax / 8]])

        draw_unlit_line(ren, [R], [np.array([1, 0, 0])], lw=lw, scale=scale)
        draw_unlit_line(ren, [G], [np.array([0, 1, 0])], lw=lw, scale=scale)
        draw_unlit_line(ren, [B], [np.array([0, 0, 1])], lw=lw, scale=scale)

    if viz_type == 'Peak':
        R = np.array([[X0, Y0, Z0], [X0 + Nmax / 8 / np.sqrt(2), Y0, Z0 + Nmax / 8 / np.sqrt(2)]])
        G = np.array([[X0, Y0, Z0], [X0, Y0 + Nmax / 8, Z0]])
        B = np.array([[X0, Y0, Z0], [X0 - Nmax / 8 / np.sqrt(2), Y0, Z0 + Nmax / 8 / np.sqrt(2)]])

        draw_unlit_line(ren, [R], [np.array([1, 0, 0])], lw=lw, scale=scale)
        draw_unlit_line(ren, [G], [np.array([0, 1, 0])], lw=lw, scale=scale)
        draw_unlit_line(ren, [B], [np.array([0, 0, 1])], lw=lw, scale=scale)


def draw_single_arrow(ren, pos, direction, color=[1, 1, 1]):
    arrow = vtk.vtkArrowSource()
    arrow.SetTipResolution(50)
    arrow.SetShaftResolution(50)

    arrowm = vtk.vtkPolyDataMapper()
    arrowm.SetInputConnection(arrow.GetOutputPort())
    arrowa = vtk.vtkActor()
    arrowa.SetMapper(arrowm)
    arrowa.GetProperty().SetColor(color)
    arrowa.SetScale(np.linalg.norm(direction))

    tp = util.xyz2tp(*direction)
    arrowa.RotateWXYZ(-90, 0, 1, 0)  # Align with Z axis
    arrowa.RotateWXYZ(np.rad2deg(tp[0]), 0, 1, 0)
    arrowa.RotateWXYZ(np.rad2deg(tp[1]), 0, 0, 1)
    arrowa.SetPosition(*pos)

    arrowa.GetProperty().SetLighting(0)
    ren.AddActor(arrowa)


def add_text(ren, text, x, y, mag, va='center', ha='center'):
    textProperty = vtk.vtkTextProperty()
    textProperty.SetFontSize(25 * mag)
    textProperty.SetFontFamilyToArial()
    textProperty.BoldOn()

    if ha == 'right':
        textProperty.SetJustificationToRight()
    elif ha == 'left':
        textProperty.SetJustificationToLeft()
    else:
        textProperty.SetJustificationToCentered()
    if va == 'top':
        textProperty.SetVerticalJustificationToTop()
    elif va == 'bottom':
        textProperty.SetVerticalJustificationToBottom()
    else:
        textProperty.SetVerticalJustificationToCentered()

    textmapper = vtk.vtkTextMapper()
    textmapper.SetTextProperty(textProperty)
    textmapper.SetInput(text)

    textactor = vtk.vtkActor2D()
    textactor.SetMapper(textmapper)
    textactor.SetPosition(500 * mag * x, 500 * mag * y)
    ren.AddActor(textactor)


def plot_den_gfa_histogram(density, gfa, filename):
    ## Plot GFA-density histogram
    hist, xedges, yedges = np.histogram2d(density, gfa, bins=np.linspace(0, 1, num=51))

    f = plt.figure(figsize=(4.5, 4))
    ax = f.add_axes([0.17, 0.02, 0.72, 0.79])
    axcolor = f.add_axes([0.90, 0.02, 0.03, 0.79])

    # ax.imshow(hist, vmin=0, vmax=np.max(hist), cmap='gray', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation=None, origin='lower')
    hist[hist == 0] = 0.01
    im = ax.imshow(hist, norm=LogNorm(vmin=0.1, vmax=np.max(hist)), cmap='jet',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation=None, origin='lower')

    t = 10 ** np.array([0, 1, 2, 3, 4, 5, 6, 7])
    f.colorbar(im, cax=axcolor, ticks=t, format='$%.f$')

    ax.set_xlabel('GFA')
    ax.set_ylabel('Density (normalized)')
    # ax.grid(color=[0.9,0.9,0.9], lw=0.2, which='both')
    ax.annotate('\# Voxels', xy=(1, 1), xytext=(1.12, 1.05), textcoords='axes fraction', xycoords='axes fraction',
                ha='center', va='center', rotation=0)

    util.mkdir(filename)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_histogram(data, filename, bin_n=26, color=[1, 0, 0], ymax=5000,
                   min_line=None, max_line=None, xlabel='GFA', xlim=[0, 1]):
    f, ax = plt.subplots(1, 1, figsize=(4, 1.5))

    ## Plot GFA-density histogram
    ax.hist(data, bins=np.linspace(xlim[0], xlim[1], num=bin_n), color=color)

    if min_line is not None:
        ax.plot([min_line, min_line], [0, ymax], '--k')
    if max_line is not None:
        ax.plot([max_line, max_line], [0, ymax], '--k')

    ax.set_xlim(xlim)
    ax.set_ylim([0, ymax])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('# Voxels')

    util.mkdir(filename)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_histogram_list(hist_list, filename, bin_n=30, color=[1, 0, 0],
                        ymax=20000, yspacing=750, xlabel='GFA'):
    f, ax = plt.subplots(1, 1, figsize=(4, 8))

    ## Plot list of histograms appropriately spaced
    for i, hist in enumerate(hist_list):
        line = ax.hist(hist, bins=np.linspace(0, 1, num=bin_n), histtype='step', bottom=yspacing * i)
        line[-1][0].set_clip_on(False)

    # Set labels
    label_pos = np.arange(0, ymax, yspacing)
    ax.set_yticks(label_pos)
    labels = np.shape(label_pos)[0] * ['']
    labels[0] = '0'
    labels[1] = str(yspacing)
    ax.set_yticklabels(labels)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, ymax])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('\# Voxels')
    ax.yaxis.set_label_coords(-0.175, 0.0)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    all_gfa = np.concatenate(hist_list)
    if all_gfa.size:  # if not empty
        ax.annotate('$\mu$ = ' + '{:.2f}'.format(np.mean(all_gfa)), xy=(1, 1), xytext=(0.9, 1.0),
                    textcoords='axes fraction', xycoords='axes fraction', ha='right', va='center', rotation=0)
        ax.annotate('$\sigma$ = ' + '{:.2f}'.format(np.std(all_gfa)), xy=(1, 1), xytext=(0.9, 0.98),
                    textcoords='axes fraction', xycoords='axes fraction', ha='right', va='center', rotation=0)

    util.mkdir(filename)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
