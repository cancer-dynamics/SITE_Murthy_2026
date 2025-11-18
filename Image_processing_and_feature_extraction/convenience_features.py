import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import skimage.filters
import imageprep as imprep
import glob, re, os
from xarray import DataArray
import scipy
import math 
from IPython.display import clear_output
from PIL import Image
def featSize(regionmask, intensity):
    size = np.sum(regionmask)
    return size

def meanIntensity(regionmask, intensity):
    mean_intensity = np.mean(intensity[regionmask])
    return mean_intensity

def totalIntensity(regionmask, intensity):
    sum_intensity = np.sum(intensity[regionmask])
    return sum_intensity

def show_3dseg_zproj(fig,
                     im3d,
                     labels,
                     fmap=None,
                     fmapscale=None,
                     labels2=None,
                     fmsk=None,
                     zscale=1.0,
                     title=None):
    """
    Three aligned projections (XY, XZ, YZ) for volumes shaped (Z, X, Y).
    Uses 'extent' with zscale (µm_z / µm_xy) to preserve geometry.
    Returns (ax_xy, ax_xz, ax_yz).  All panels share the same visual height.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    im3d   = np.asarray(im3d)
    labels = np.asarray(labels)
    Z, X, Y = im3d.shape
    zscale = 1.0 if (zscale is None) else float(zscale)

    # Max projections
    zproj  = np.max(im3d, axis=0)   # (X, Y) -> XY
    xz     = np.max(im3d, axis=2)   # (Z, X) -> XZ
    yz     = np.max(im3d, axis=1)   # (Z, Y) -> YZ

    Lz  = np.max(labels, axis=0)    # (X, Y)
    Lxz = np.max(labels, axis=2)    # (Z, X)
    Lyz = np.max(labels, axis=1)    # (Z, Y)

    if fmap is not None:
        fmap = np.asarray(fmap)
        Fmap_z  = np.max(fmap, axis=0)
        Fmap_xz = np.max(fmap, axis=2)
        Fmap_yz = np.max(fmap, axis=1)

    if labels2 is not None:
        labels2 = np.asarray(labels2)
        L2z  = np.max(labels2, axis=0)
        L2xz = np.max(labels2, axis=2)
        L2yz = np.max(labels2, axis=1)

    if fmsk is not None:
        fmsk = np.asarray(fmsk)
        Fz  = np.max(fmsk, axis=0)
        Fxz = np.max(fmsk, axis=2)
        Fyz = np.max(fmsk, axis=1)

    plt.figure(fig.number); plt.clf()
    if title:
        plt.suptitle(title)

    # ---------- XY ----------
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(zproj.T, cmap=plt.cm.gray_r, origin='lower',
               extent=(0, Y, 0, X))
    lv = np.unique(Lz)[1:]  # skip 0
    if lv.size:
        ax1.contour(Lz.T, levels=lv, colors='blue', linewidths=1,
                    origin='lower', extent=(0, Y, 0, X))
    if fmap is not None:
        im = ax1.imshow(Fmap_z.T, cmap=plt.cm.YlOrRd, origin='lower',
                        extent=(0, Y, 0, X), alpha=0.33)
        if fmapscale is not None:
            im.set_clim(*fmapscale)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_axis_off()

    # ---------- XZ ----------
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(xz.T, cmap=plt.cm.gray_r, origin='lower',
               extent=(0, Z * zscale, 0, X))
    lv = np.unique(Lxz)[1:]
    if lv.size:
        ax2.contour(Lxz.T, levels=lv, colors='blue', linewidths=1,
                    origin='lower', extent=(0, Z * zscale, 0, X))
    if fmap is not None:
        im = ax2.imshow(Fmap_xz.T, cmap=plt.cm.YlOrRd, origin='lower',
                        extent=(0, Z * zscale, 0, X), alpha=0.33)
        if fmapscale is not None:
            im.set_clim(*fmapscale)
    ax2.set_aspect('equal', adjustable='box')   # <- same visual height as XY
    ax2.set_axis_off()

    # ---------- YZ ----------
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(yz.T, cmap=plt.cm.gray_r, origin='lower',
               extent=(0, Z * zscale, 0, Y))
    lv = np.unique(Lyz)[1:]
    if lv.size:
        ax3.contour(Lyz.T, levels=lv, colors='blue', linewidths=1,
                    origin='lower', extent=(0, Z * zscale, 0, Y))
    if fmap is not None:
        im = ax3.imshow(Fmap_yz.T, cmap=plt.cm.YlOrRd, origin='lower',
                        extent=(0, Z * zscale, 0, Y), alpha=0.33)
        if fmapscale is not None:
            im.set_clim(*fmapscale)
    ax3.set_aspect('equal', adjustable='box')   # <- same visual height as XY
    ax3.set_axis_off()

    plt.tight_layout()
    return ax1, ax2, ax3

def show_3dseg_zproj_segmentation(fig,im3d,labels,fmap=None,fmapscale=None,labels2=None,fmsk=None,zscale=None):
    if zscale is not None:
        im3d=skimage.transform.rescale(im3d,(zscale,1,1),anti_aliasing=False)
        labels=skimage.transform.rescale(labels,(zscale,1,1),order=0)
        if labels2 is not None:
            labels2=skimage.transform.rescale(labels2,(zscale,1,1),order=0)
        if fmap is not None:
            fmap=skimage.transform.rescale(fmap,(zscale,1,1),anti_aliasing=False)
    zproj=np.sum(im3d,axis=0)
    xproj=np.sum(im3d,axis=1)
    yproj=np.sum(im3d,axis=2)
    labels_zproj=np.max(labels,axis=0)
    labels_xproj=np.max(labels,axis=1)
    labels_yproj=np.max(labels,axis=2)
    plt.subplot(1,3,1)
    plt.imshow(zproj,cmap=plt.cm.gray_r)
    plt.contour(labels_zproj,levels=np.arange(np.max(labels_zproj)),colors='blue',linewidths=1)
    if fmap is not None:
        #plt.contour(np.max(fmap,axis=0),levels=np.linspace(np.min(fmap),np.max(fmap),21),cmap=plt.cm.PiYG_r)
        plt.imshow(np.max(fmap,axis=0),cmap=plt.cm.YlOrRd,alpha=.33,clim=fmapscale)
    if labels2 is not None:
        labels2_zproj=np.max(labels2,axis=0)
        plt.contour(labels2_zproj,levels=np.arange(np.max(labels2_zproj)),colors='green',linewidths=1)
    if fmsk is not None:
        fmsk_zproj=np.max(fmsk,axis=0)
        plt.contour(fmsk_zproj,levels=np.arange(np.max(fmsk_zproj)),colors='red',linewidths=1)    
    plt.axis('equal');plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(xproj,cmap=plt.cm.gray_r)
    plt.contour(labels_xproj,levels=np.arange(np.max(labels_xproj)),colors='blue',linewidths=1)
    if fmap is not None:
        #plt.contour(np.max(fmap,axis=0),levels=np.linspace(np.min(fmap),np.max(fmap),21),cmap=plt.cm.PiYG_r)
        plt.imshow(np.max(fmap,axis=1),cmap=plt.cm.YlOrRd,alpha=.33,clim=fmapscale)
    if labels2 is not None:
        labels2_xproj=np.max(labels2,axis=1)
        plt.contour(labels2_xproj,levels=np.arange(np.max(labels2_xproj)),colors='green',linewidths=1)
    if fmsk is not None:
        fmsk_xproj=np.max(fmsk,axis=1)
        plt.contour(fmsk_xproj,levels=np.arange(np.max(fmsk_xproj)),colors='red',linewidths=1)    
    plt.axis('equal');plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(yproj,cmap=plt.cm.gray_r)
    plt.contour(labels_yproj,levels=np.arange(np.max(labels_yproj)),colors='blue',linewidths=1)
    if fmap is not None:
        #plt.contour(np.max(fmap,axis=0),levels=np.linspace(np.min(fmap),np.max(fmap),21),cmap=plt.cm.PiYG_r)
        plt.imshow(np.max(fmap,axis=2),cmap=plt.cm.YlOrRd,alpha=.33,clim=fmapscale)
    if labels2 is not None:
        labels2_yproj=np.max(labels2,axis=2)
        plt.contour(labels2_yproj,levels=np.arange(np.max(labels2_yproj)),colors='green',linewidths=1)
    if fmsk is not None:
        fmsk_yproj=np.max(fmsk,axis=2)
        plt.contour(fmsk_yproj,levels=np.arange(np.max(fmsk_yproj)),colors='red',linewidths=1)    
    plt.axis('equal');plt.axis('off')

def show_image_3channel(bf_img,nuc_img,rep_img,ax=None):
        if ax is None:
            ax=plt.gca()
        ax.imshow(imprep.znorm(bf_img),cmap=plt.cm.binary,clim=(-5,5))
        cs=ax.contour(imprep.znorm(nuc_img),cmap=plt.cm.BuPu,levels=np.linspace(0,np.percentile(imprep.znorm(nuc_img),99),7),alpha=.5,linewidths=.5)
        cs.cmap.set_over('purple')
        cs=ax.contour(imprep.znorm(rep_img),cmap=plt.cm.YlOrBr_r,levels=np.linspace(0,np.percentile(imprep.znorm(rep_img),99),7),alpha=.25,linewidths=.5)
        cs.cmap.set_over('yellow')
        return ax

def rescale_to_int(img,maxint=2**16-1,dtype=np.uint16):
    img=maxint*((img-np.min(img))/np.max(img-np.min(img)))
    return img.astype(dtype)

def log_ztransform(img):
    img=np.log(imprep.znorm(img)+1.-np.min(imprep.znorm(img)))
    return img

def nuc_viewtransform(im3d_nuc,zcut_nuc=2.0,rsmooth=1.0,zscale=5.0):
    im3d_nuc=skimage.filters.gaussian(im3d_nuc,sigma=[rsmooth/zscale,rsmooth,rsmooth])
    im3d_nuc=imprep.znorm(im3d_nuc)
    im3d_nuc[im3d_nuc<zcut_nuc]=0
    im3d_nuc=rescale_to_int(log_ztransform(im3d_nuc))
    return im3d_nuc

def bf_viewtransform(im3d_bf,rsmooth=10.0,zscale=5.0,zcut=2.0,remove_background=False):
    im3d_bf=skimage.filters.difference_of_gaussians(im3d_bf,.5,1.5)
    if remove_background:
        im3d_bf_smth=skimage.filters.gaussian(np.abs(im3d_bf),sigma=[rsmooth/zscale,rsmooth,rsmooth])
        msk_bf=imprep.znorm(im3d_bf_smth)>zcut
        msk_bf=binary_fill_holes_2dstack(msk_bf)
    im3d_bf=rescale_to_int(log_ztransform(im3d_bf))
    im3d_bf=np.max(im3d_bf)-im3d_bf
    if remove_background:
        im3d_bf[np.logical_not(msk_bf)]=0
    return im3d_bf

def rep_viewtransform(im3d_rep,rsmooth=1.0,zscale=5.0):
    im3d_rep=skimage.filters.gaussian(im3d_rep,sigma=[rsmooth/zscale,rsmooth,rsmooth])
    im3d_rep=rescale_to_int(log_ztransform(im3d_rep))
    return im3d_rep

def binary_fill_holes_2dstack(im3d):
    for iz in range(im3d.shape[0]):
        im3d[iz,...]=ndimage.binary_fill_holes(im3d[iz,...])
    return im3d

def get_indices(modelName, pattern, ext):
    files = glob.glob(pattern)
    out = {}
    ext = re.escape(ext.lstrip('.'))  # accept "tif" or ".tif"
    for f in files:
        base = os.path.basename(f)
        m = re.search(rf"{re.escape(modelName)}_XY(\d+)_roi(\d+)\.{ext}$",
                      base, re.IGNORECASE)
        if m:
            xy, roi = int(m.group(1)), int(m.group(2))
            out[(xy, roi)] = base   # keep basename (matches your printing)
            # If you prefer full path, use: out[(xy, roi)] = f
    return out

def natural_key(path):
    """Sorts ..._roi2.tif before ..._roi10.tif by extracting ints from basename."""
    base = os.path.basename(path)
    nums = re.findall(r'\d+', base)
    return [int(n) for n in nums] if nums else [base.lower()]

def get_surf_fmask_lung_cancer(img_ilastik,ilastik_model,cdict_ilp,cdict_surf,histp=None,sigma=5,thresh=0.2,return_prediction=False):
    image = DataArray(img_ilastik, dims=("y", "x", "c"))
    prediction = ilastik_model.predict(image)
    prediction=prediction.to_numpy()
    for ichannel in range(prediction.shape[2]):
        pred=prediction[...,ichannel]
        if sigma is not None:
            pred=scipy.ndimage.gaussian_filter(pred,sigma=sigma)
        if histp is not None:
            plow, phigh = np.percentile(pred, (histp[0], histp[1]))
            pred=(pred-plow)/(phigh-plow)
        prediction[...,ichannel]=pred
    max_pred=np.argmax(prediction,axis=2)
    fmasks=np.zeros((img_ilastik.shape[0],img_ilastik.shape[1],2)).astype(bool)
    fmasks[...,0]=(prediction[...,cdict_ilp['primary']]-prediction[...,cdict_ilp['matrix']]>thresh)
    fmasks[...,1]=(prediction[...,cdict_ilp['cancer']]+prediction[...,cdict_ilp['cancer']]-prediction[...,cdict_ilp['primary']]-prediction[...,cdict_ilp['matrix']]>thresh)
    if return_prediction:
        return prediction,fmasks
    else:
        return fmasks
    return fmasks

def show_3dseg_proj(fig,im3d,labels,surfaces=None,surf_cmaps=None,labels2=None,fmsk=None,zscale=None,label_color='black',label2_color='red',img_cmap=plt.cm.gray_r):
    if zscale is not None:
        im3d=skimage.transform.rescale(im3d,(zscale,1,1),anti_aliasing=False)
        labels=skimage.transform.rescale(labels,(zscale,1,1),order=0)
        for isurf in range(len(surfaces)):
            surfaces[isurf]=skimage.transform.rescale(surfaces[isurf],(zscale,1,1),order=0)
        if labels2 is not None:
            labels2=skimage.transform.rescale(labels2,(zscale,1,1),order=0)
    if surf_cmaps is None and surfaces is not None:
        print(len(surfaces))
        surf_cmaps=[plt.cm.Blues,plt.cm.Greens,plt.cm.Reds,plt.cm.Purples,plt.cm.Oranges,plt.cm.Blues,plt.cm.Greens]
    if label_color is None:
        label_color='black'
    if label2_color is None:
        label2_color='red'
    if img_cmap is None:
        img_cmap=plt.cm.gray_r
    zproj=np.sum(im3d,axis=0)
    xproj=np.sum(im3d,axis=1)
    yproj=np.sum(im3d,axis=2)
    labels_zproj=np.max(labels,axis=0)
    labels_xproj=np.max(labels,axis=1)
    labels_yproj=np.max(labels,axis=2)
    plt.subplot(1,3,1)
    plt.imshow(zproj,cmap=img_cmap)
    plt.contour(labels_zproj,levels=np.arange(np.max(labels_zproj)),colors=label_color,linewidths=1)
    if surfaces is not None:
        for isurf in range(len(surfaces)):
            surf_viz=np.sum(surfaces[isurf],axis=0)
            plt.imshow(np.ma.masked_where(np.logical_not(surf_viz>0),surf_viz),alpha=.5,cmap=surf_cmaps[isurf])#,clim=(0,1))
    if labels2 is not None:
        labels2_zproj=np.max(labels2,axis=0)
        plt.contour(labels2_zproj,levels=np.arange(np.max(labels2_zproj)),colors=label2_color,linewidths=1)  
    plt.axis('equal');plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(xproj,cmap=img_cmap)
    plt.contour(labels_xproj,levels=np.arange(np.max(labels_xproj)),colors=label_color,linewidths=1)
    if surfaces is not None:
        for isurf in range(len(surfaces)):
            surf_viz=np.sum(surfaces[isurf],axis=1)
            plt.imshow(np.ma.masked_where(np.logical_not(surf_viz>0),surf_viz),alpha=.5,cmap=surf_cmaps[isurf])#,clim=(0,1))
    if labels2 is not None:
        labels2_xproj=np.max(labels2,axis=1)
        plt.contour(labels2_xproj,levels=np.arange(np.max(labels2_xproj)),colors=label2_color,linewidths=1)
    plt.axis('equal');plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(yproj,cmap=img_cmap)
    plt.contour(labels_yproj,levels=np.arange(np.max(labels_yproj)),colors=label_color,linewidths=1)
    if labels2 is not None:
        labels2_yproj=np.max(labels2,axis=2)
        plt.contour(labels2_yproj,levels=np.arange(np.max(labels2_yproj)),colors=label2_color,linewidths=1)
    if surfaces is not None:
        for isurf in range(len(surfaces)):
            surf_viz=np.sum(surfaces[isurf],axis=2)
            plt.imshow(np.ma.masked_where(np.logical_not(surf_viz>0),surf_viz),alpha=.5,cmap=surf_cmaps[isurf])#,clim=(0,1)) 
    plt.axis('equal');plt.axis('off')

def xy_sort_key(x): 
    m = re.search(r'XY(\d+)', x)
    return int(m.group(1)) if m else float('inf')

def get_tifs_for_png(png_path):
    """
    Return all .tif files matching the same XY and ROI as png_path.
    Matches both *_XY8_roi2.tif and *_XY8_roi2_anything.tif
    """
    fn = os.path.basename(png_path).split('_')
    xy  = next(p for p in fn if p.startswith('XY'))
    roi = next(p for p in fn if p.startswith('roi'))
    # note: no underscore before the wildcard
    pat = os.path.join(image_dir, f"*_{xy}_{roi}*.tif")
    return glob.glob(pat)

def display_thumbnails(xy, selected, rejected, xy_groups):
    entries = xy_groups[xy]
    n = len(entries)
    cols = 8
    rows = math.ceil(n/cols)
    clear_output(wait=True)
    if rows > 4:
        print(f"{xy}: {n} ROIs (too many—thumbnails skipped)\n")
        return
    fig, axs = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axs = axs.flatten()
    # blank out extra axes
    for ax in axs[n:]:
        ax.axis('off')
    # draw each thumbnail with a colored border
    for i,(png,roi) in enumerate(entries):
        ax = axs[i]
        im = Image.open(png)
        ax.imshow(im, cmap='gray')
        ax.set_title(roi, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('lightgray')
            spine.set_linewidth(1)
        uid = (xy, roi)
        if uid in selected:
            col='green'
        elif uid in rejected:
            col='red'
        else:
            col=None
        if col:
            for spine in ax.spines.values():
                spine.set_edgecolor(col)
                spine.set_linewidth(4)
    plt.tight_layout()
    plt.show()

def prompt_roi(png, xy, roi, selected, rejected, xy_groups):
    """
    Returns:
      'y' (keep), 'n' (reject), 's' (skip XY), 'q' (quit)
    """
    display_thumbnails(xy, selected, rejected, xy_groups)
    # show large image
    clear_output(wait=True)
    display_thumbnails(xy, selected, rejected, xy_groups) 
    im = Image.open(png)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(im, cmap='gray')
    ax.set_title(f"{xy} {roi}")
    ax.set_xticks([]); ax.set_yticks([])
    plt.show()

    valid = {'y','n','s','q'}
    prompt = ("[y=keep | n=reject | s=skip-this-XY | q=quit] → ").strip()
    while True:
        resp = input(f"{xy} {roi} {prompt}").lower().strip()
        if resp in valid:
            return resp
        print("↳ please type y, n, s, or q")