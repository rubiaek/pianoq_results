def my_mesh(X, Y, C, ax=None, clim=None, c_label=None):
    if ax is None:
        fig, ax = plt.subplots()
    if len(X) >= 2 and len(Y) >= 2:
        dx = (X[1] - X[0]) / 2
        dy = (Y[1] - Y[0]) / 2
        extent = (X[0]-dx, X[-1]+dx, Y[0]-dy, Y[-1]+dy)
        im = ax.imshow(C, extent=extent)
    else:
        im = ax.imshow(C)
    if clim:
        im.set_clim(*clim)
    cbar = ax.figure.colorbar(im, ax=ax)
    if c_label:
        cbar.set_label(c_label)
    return im
