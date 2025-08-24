import logging
import numpy as np
from PFM import util
from PFM import visualization as viz_pkg
# import vtk
import vtkmodules.all as vtk
from dipy.viz import window
from tqdm import tqdm
import subprocess

log = logging.getLogger('log')


def visualize(self, out_path='out/', viz_type=['Peak'], scalemap=None, scale=1, interact=True, mag=1, peak_scale=1,
              bw_invert=False, mask=None, skip_n=1, density_max=1, gamma=1, titles=True, titles_txt=None, scalebar=True,
              outer_box=True, profiles=None,
              color_axes=True, color_axes_lw_rat=1, camtilt=False, det_axes=None, azimuth=0, elevation=0, n_frames=18,
              select_frame=None, video=False, rois=None, rois_color=None, lines=None):
    log.info('Preparing to render ' + out_path)

    util.mkdir(out_path)

    if scalemap is None:
        scalemap = viz_pkg.ScaleMap(min=np.min(self.f[..., 0]), max=np.max(self.f[..., 0]))

    renWin = vtk.vtkRenderWindow()

    if not interact:
        renWin.SetOffScreenRendering(1)

    cols = len(viz_type)
    renWin.SetSize(np.int(500 * mag * cols), np.int(500 * mag))

    if not bw_invert:
        bg_color = [0, 0, 0]
        line_color = np.array([1, 1, 1])
        line_bcolor = np.array([0, 0, 0])
    else:
        bg_color = [1, 1, 1]
        line_color = np.array([0, 0, 0])
        line_bcolor = np.array([1, 1, 1])

    rens = []
    for col in range(cols):
        ren = window.Scene()
        rens.append(ren)
        ren.background(bg_color)

        ren.SetViewport(col / cols, 0, (col + 1) / cols, 1)
        renWin.AddRenderer(ren)
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        if mask is None:
            mask = np.ones((self.X, self.Y, self.Z), dtype=np.bool)

        data = self.f
        skip_mask = np.zeros(mask.shape, dtype=np.bool)
        skip_mask[::skip_n, ::skip_n, ::skip_n] = 1
        my_mask = np.logical_and(mask, skip_mask)
        if my_mask.sum() == 0:
            self.f[0, 0, 0, :] = 0
            my_mask[0, 0, 0] = True

        scale = scale
        scalemap = scalemap

        # Add visuals to renderer
        if mask.sum() != 0:
            if viz_type[col] == "ODF":
                renWin.SetMultiSamples(4)
                log.info('Rendering ' + str(np.sum(my_mask)) + ' ODFs')
                fodf_spheres = viz_pkg.odf_sparse(data, self.Binv, sphere=self.sphere,
                                                  scale=skip_n * scale * 0.5, norm=False,
                                                  colormap='bwr', mask=my_mask,
                                                  global_cm=True, scalemap=scalemap,
                                                  odf_sphere=False, flat=False, normalize=True)
                ren.add(fodf_spheres)

            if viz_type[col] == "Peak":
                renWin.SetMultiSamples(4)
                log.info('Rendering ' + str(np.sum(my_mask)) + ' peaks')
                fodf_peaks = viz_pkg.peak_slicer_sparse(data, self.Binv, self.sphere.vertices, peak_scale=peak_scale,
                                                        linewidth=0.1, scale=skip_n * scale * 0.5, colors=None,
                                                        mask=my_mask, scalemap=scalemap, normalize=True)
                ren.add(fodf_peaks)

            if viz_type[col] == "Peak Ball":
                renWin.SetMultiSamples(4)
                log.info('Rendering ' + str(np.sum(my_mask)) + ' peaks')
                fodf_peaks = viz_pkg.peak_slicer_sparse(data, self.Binv, self.sphere.vertices, peak_scale=peak_scale,
                                                        linewidth=0.1, scale=skip_n * scale * 0.5, colors=None,
                                                        mask=my_mask, scalemap=scalemap, normalize=True, balls=True)
                ren.add(fodf_peaks)

            if viz_type[col] == "Peak_MP Ball":
                renWin.SetMultiSamples(4)
                log.info('Rendering ' + str(np.sum(my_mask)) + ' peaks')
                fodf_peaks = viz_pkg.peak_mp_slicer_sparse(data, self.Binv, self.sphere.vertices, peak_scale=peak_scale,
                                                           linewidth=0.1, scale=skip_n * scale * 0.5, colors=None,
                                                           mask=my_mask, scalemap=scalemap, normalize=True, balls=True)
                ren.add(fodf_peaks)

            if viz_type[col] == "Peak_MP":
                renWin.SetMultiSamples(4)
                log.info('Rendering ' + str(np.sum(my_mask)) + ' peaks')
                fodf_peaks = viz_pkg.peak_mp_slicer_sparse(data, self.Binv, self.sphere.vertices, peak_scale=peak_scale,
                                                           linewidth=0.1, scale=skip_n * scale * 0.5, colors=None,
                                                           mask=my_mask, scalemap=scalemap, normalize=True)
                ren.add(fodf_peaks)

            if viz_type[col] == "Density":
                renWin.SetMultiSamples(0)  # Must be zero for smooth
                log.info('Rendering density')

                gamma_corr = np.where(data[..., 0] > 0, data[..., 0] ** gamma, data[..., 0])
                scalemap.max = density_max * scalemap.max ** gamma
                volume = viz_pkg.density_slicer(gamma_corr, scalemap)
                ren.add(volume)

            if viz_type[col] == "Ellipsoid":
                renWin.SetMultiSamples(4)
                log.info('Warning: scaling is not implemented for ellipsoids')
                log.info('Rendering ' + str(np.sum(my_mask)) + ' ellipsoids')
                fodf_peaks = viz_pkg.tensor_slicer_sparse(data,
                                                          sphere=self.sphere,
                                                          scale=skip_n * scale * 0.5,
                                                          mask=my_mask)
                ren.add(fodf_peaks)

            if viz_type[col] == "ODF Sphere":
                renWin.SetMultiSamples(4)
                log.info('Rendering ' + str(np.sum(my_mask)) + ' ODFs')
                fodf_spheres = viz_pkg.odf_sparse(data, self.Binv, sphere=self.sphere,
                                                  scale=skip_n * scale * 0.5, norm=False,
                                                  colormap='bwr', mask=my_mask,
                                                  global_cm=True, scalemap=scalemap,
                                                  odf_sphere=True, flat=False)
                ren.add(fodf_spheres)

        # Titles
        if titles:
            if titles_txt is None:
                viz_pkg.add_text(ren, viz_type[col], 0.5, 0.96, mag)
            else:
                viz_pkg.add_text(ren, titles_txt[col], 0.5, 0.96, mag)

        X = np.float(data.shape[0])
        Y = np.float(data.shape[1])
        Z = np.float(data.shape[2])

        # Scale bar
        if scalebar:
            yscale = 1e-3 * self.vox_dim[1] * data.shape[1]
            yscale_label = '{:.2g}'.format(yscale) + ' um'
            viz_pkg.add_text(ren, yscale_label, 0.5, 0.03, mag)
            viz_pkg.draw_scale_bar(ren, X, Y, Z, [1, 1, 1])

        # Draw boxes
        if outer_box:
            viz_pkg.draw_outer_box(ren, np.array([[0, 0, 0], [X, Y, Z]]) - 0.5, line_color)

        # Add colored axes
        if color_axes:
            viz_pkg.draw_axes(ren, np.array([[0, 0, 0], [X, Y, Z]]) - 0.5, viz_type[col], color_axes_lw_rat)

        if lines is not None:
            for line in lines:
                viz_pkg.draw_unlit_line(ren, line, [line_color], lw=0.3, scale=1.0)

        # Setup cameras
        Rmax = np.linalg.norm([Z / 2, X / 2, Y / 2])
        Rcam_rad = Rmax / np.tan(np.pi / 12)
        Ntmax = np.max([X, Y])
        ZZ = Z
        if ZZ > Ntmax:
            Rcam_edge = np.max([X / 2, Y / 2])
        else:
            Rcam_edge = np.min([X / 2, Y / 2])
        Rcam = Rcam_edge + Rcam_rad

        cam = ren.GetActiveCamera()
        if camtilt:
            cam.SetPosition(((X - 1) / 2, (Y - 1) / 2, (Z - 1) / 2 + Rcam))
            cam.SetViewUp((-1, 0, 1))
        else:
            cam.SetPosition(((X - 1) / 2 + Rcam, (Y - 1) / 2, (Z - 1) / 2))
            cam.SetViewUp((0, 0, 1))
        cam.SetFocalPoint(((X - 1) / 2, (Y - 1) / 2, (Z - 1) / 2))

        # Show det axes
        if det_axes is not None:
            max_dim = np.max((X, Z))
            length = max_dim / 2
            for idx in range(det_axes.shape[0]):
                det_z = det_axes[idx, 2, :]
                point = list(det_z * length + np.array([X / 2, Y / 2, Z / 2]))
                viz_pkg.draw_unlit_line(ren, [np.array([point, [X / 2, Y / 2, Z / 2]])],
                                        3 * [line_color], lw=max_dim / 250, scale=1.0)

        # set showing azimuth and elevation
        ren.azimuth(azimuth)
        ren.elevation(elevation)

    # for x in range(self.X):
    #     for y in range(self.Y):
    #         for z in range(self.Z):
    #             pos = [x, y, z]
    #             dir = [0, -1, 0]
    #             color = [0, 1, 0]
    #             viz_pkg.draw_single_arrow(ren, pos=pos, direction=dir, color=color)

    if rois is not None:
        for idx, roi in enumerate(rois):
            roi = [[roi[0][0], roi[1][0], roi[2][0]], [roi[0][1], roi[1][1], roi[2][1]]]
            maxROI = np.max([roi[1][0] - roi[0][0], roi[1][1] - roi[0][1], roi[1][2] - roi[0][2]])
            maxXYZ = np.max([self.X, self.Y, self.Z])

            if rois_color is None:
                if bw_invert == False:
                    viz_pkg.draw_outer_box(ren, roi, [0, 1, 1], lw=0.2 * maxXYZ / maxROI)
                else:
                    viz_pkg.draw_outer_box(ren, roi, [1, 0, 1], lw=0.2 * maxXYZ / maxROI * 4)
                # viz_pkg.draw_axes(ren, roi, lw=0.3 * maxXYZ / maxROI)
            else:
                viz_pkg.draw_outer_box(ren, roi, rois_color[idx], lw=0.2 * maxXYZ / maxROI)

    if profiles is not None:
        for idx, profile_dict in enumerate(profiles):
            profile = profile_dict['profile']
            color = profile_dict['color']

            viz_pkg.draw_profile(ren, X, Y, Z, profile, color, lw=0.5)

    # Setup writer
    writer = vtk.vtkTIFFWriter()

    # Execute renders
    az = 90
    naz = np.ceil(360 / n_frames)
    log.info('Rendering ' + out_path)

    # Rendering for movies
    for j, ren in enumerate(rens):
        ren.zoom(1.3)

    for i in tqdm(range(n_frames)):
        for j, ren in enumerate(rens):
            ren.zoom(1)
            ren.azimuth(az)
            ren.reset_clipping_range()

        renderLarge = vtk.vtkRenderLargeImage()
        renderLarge.SetMagnification(1)
        renderLarge.SetInput(ren)
        renderLarge.Update()
        writer.SetInputConnection(renderLarge.GetOutputPort())

        az = naz

        if select_frame is None:
            if n_frames != 1:
                writer.SetFileName(out_path + str(i).zfill(3) + '.tif')
            else:
                writer.SetFileName(out_path + '.tif')

        if select_frame is not None:
            if i != select_frame:
                continue
            else:
                writer.SetFileName(out_path + '.tif')

        writer.Write()

    # Interactive
    if interact:
        window.show(ren)

    # Generate video (requires ffmpeg)
    if video:
        log.info('Generating video from frames')
        fps = np.ceil(n_frames / 12)
        subprocess.call(['ffmpeg', '-nostdin', '-y', '-framerate', str(fps),
                         '-loglevel', 'panic', '-i', out_path + '%03d' + '.tif',
                         '-pix_fmt', 'yuvj420p', '-vcodec', 'mjpeg',
                         out_path[:-1] + '.avi'])
