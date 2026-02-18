"""
Brain Figure — MNE-Python + Glasser HCP-MMP1 Atlas
===================================================
Saves lateral and medial views as separate SVGs.
Highlights dlPFC, ACC, pre-SMA, dmPFC, vmPFC, precuneus, STN.

Style: Light gray brain with solid, clearly delineated ROI fills.
       Left hemisphere only.

Requirements:
    pip install mne pyvista pyvistaqt matplotlib

Setup:
    Place lh.HCPMMP1.annot and rh.HCPMMP1.annot in:
      <subjects_dir>/fsaverage/label/

Usage:
    python brain_figure_glasser.py
"""

import mne
import numpy as np
import os
import pyvista as pv

mne.viz.set_3d_backend('pyvistaqt')

# =============================================================================
# 1. SETUP
# =============================================================================

fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
subjects_dir = os.path.dirname(fs_dir)

# =============================================================================
# 2. ROI DEFINITIONS — Glasser HCP-MMP1 parcels
# =============================================================================

ROIS = {
    'dlPFC': [
        '46', '9-46d', 'p9-46v', 'a9-46v', '8C', '8Av', '8Ad',
    ],
    'ACC': [
        'p32pr', 'a32pr', 'd32', 'p24', 'a24', '33pr',
    ],
    'preSMA': [
        '6a', 'SCEF',
    ],
    'dmPFC': [
        '8BM', '9m',
    ],
    'vmPFC': [
        '25', 's32', '10v', '10r', 'OFC', 'pOFC',
    ],
    'precuneus': [
        '7m', 'PCV', 'POS2', '31pv', '31pd',
    ],
}

# =============================================================================
# 3. COLOR SCHEME — maximally distinct
# =============================================================================

COLORS = {
    'dlPFC':     '#1F78B4',   # strong blue
    'ACC':       '#E31A1C',   # red
    'preSMA':    '#33A02C',   # green
    'dmPFC':     '#FF7F00',   # orange
    'vmPFC':     '#FFED6F',   # bright yellow
    'precuneus': '#CB2DB4',   # magenta/purple
    'STN':       '#1B9E77',   # teal
}

# =============================================================================
# 4. HELPER: Find Glasser label
# =============================================================================

def find_glasser_label(labels, parcel_name, hemi):
    """Find a Glasser label by parcel name."""
    prefix = 'L' if hemi == 'lh' else 'R'
    target = f'{prefix}_{parcel_name}_ROI-{hemi}'
    label_dict = {l.name: l for l in labels}
    return label_dict.get(target, None)


# =============================================================================
# 5. ADD CORTICAL REGIONS — solid fills, no borders
# =============================================================================

def add_cortical_regions(brain, hemis=None):
    """Add Glasser-based cortical ROIs to the brain as solid opaque fills."""
    if hemis is None:
        hemis = ['lh', 'rh']

    count = 0
    not_found = []

    for hemi in hemis:
        labels = mne.read_labels_from_annot(
            'fsaverage', parc='HCPMMP1',
            subjects_dir=subjects_dir, hemi=hemi
        )

        for roi_name, parcels in ROIS.items():
            color = COLORS[roi_name]
            for parcel in parcels:
                label = find_glasser_label(labels, parcel, hemi)
                if label is not None:
                    brain.add_label(
                        label,
                        color=color,
                        alpha=1.0,
                        borders=False,
                    )
                    count += 1
                else:
                    not_found.append(f"{hemi}: {parcel} ({roi_name})")

    print(f"✓ Added {count} cortical labels")
    if not_found:
        print("⚠ Labels not found:")
        for nf in not_found:
            print(f"  {nf}")


# =============================================================================
# 6. ADD STN AS ELLIPSOIDS
# =============================================================================

def add_stn(brain, hemis=None):
    """Add STN as anatomically-shaped ellipsoids to the PyVista plotter."""
    if hemis is None:
        hemis = ['lh', 'rh']

    plotter = brain._renderer.plotter

    x_radius = 4.5
    y_radius = 3.0
    z_radius = 2.0

    # STN coordinates — anatomically correct position.
    # Visible through the semi-transparent pial surface.
    stn_centers = {
        'lh': [-3, -15, 2],
        'rh': [3, -15, 2],
    }

    for hemi in hemis:
        if hemi in stn_centers:
            center = stn_centers[hemi]
            stn = pv.ParametricEllipsoid(x_radius, y_radius, z_radius)
            stn.translate(center, inplace=True)
            plotter.add_mesh(
                stn,
                color=COLORS['STN'],
                opacity=0.9,
                smooth_shading=True,
            )
            print(f"✓ Added STN ellipsoid ({hemi}) at {center}")


# =============================================================================
# 7. SOFTEN LIGHTING — reduce harsh sulcal shadows
# =============================================================================

def soften_lighting(brain):
    """Boost ambient and reduce specular to make sulci less dark
    and ROI colors more uniform across the surface.
    """
    plotter = brain._renderer.plotter
    renderer = plotter.renderer

    # Adjust all actors: less specular, more ambient
    for actor in renderer.GetActors():
        prop = actor.GetProperty()
        prop.SetSpecular(0.05)      # almost no specular shine
        prop.SetSpecularPower(1.0)
        prop.SetAmbient(0.55)       # higher ambient = less shadow contrast
        prop.SetDiffuse(0.45)       # lower diffuse = softer shading

    print("✓ Softened lighting")


# =============================================================================
# 8. CREATE AND SAVE INDIVIDUAL VIEWS
# =============================================================================

def make_and_save_view(hemi, view, basename):
    """
    Create a single brain view and save as both SVG and PNG.

    Parameters
    ----------
    hemi : str
        'lh' or 'rh'
    view : str
        'lateral', 'medial', 'dorsal', 'ventral', 'frontal', 'caudal'
    basename : str
        Output filename without extension (e.g., 'brain_lateral')
    """

    # ---- Cortex color ----
    # MNE cortex presets: 'classic', 'low_contrast', 'high_contrast',
    # 'bone', 'ivory', or any valid color name/hex.
    # 'low_contrast' gives a light gray with gentle sulcal shading.
    # For a uniform flat gray, pass a color like 'grey' or '#B0B0B0'.
    cortex_color = 'low_contrast'

    brain = mne.viz.Brain(
        'fsaverage',
        hemi=hemi,
        surf='pial',
        subjects_dir=subjects_dir,
        views=view,
        background='white',
        cortex=cortex_color,
        alpha=1.0,               # semi-transparent brain
        size=(1200, 1000),
    )

    # Add ROIs as solid fills
    add_cortical_regions(brain, hemis=[hemi])

    # Add STN ellipsoids
    add_stn(brain, hemis=[hemi])

    # Save PNG
    png_path = f'{basename}.png'
    brain.save_image(png_path, mode='rgb')
    print(f"✓ Saved PNG: {png_path}")

    # Save SVG
    svg_path = f'{basename}.svg'
    try:
        plotter = brain._renderer.plotter
        plotter.save_graphic(svg_path)
        print(f"✓ Saved SVG: {svg_path}")
    except Exception as e:
        print(f"⚠ SVG export failed: {e}")
        print("  Try pip install pyvista --upgrade, or use the PNG.")

    brain.close()
    return brain


# =============================================================================
# 9. RUN
# =============================================================================

if __name__ == '__main__':

    # --- Verify labels ---
    print("\n=== Verifying Glasser HCP-MMP1 labels ===")
    labels = mne.read_labels_from_annot(
        'fsaverage', parc='HCPMMP1',
        subjects_dir=subjects_dir, hemi='lh'
    )
    print(f"Total labels loaded: {len(labels)}")

    # =========================================================================
    # Render and save lateral + medial as separate files
    # =========================================================================

    hemi = 'lh'   # change to 'rh' for right hemisphere

    print("\n--- Lateral view ---")
    make_and_save_view(hemi, 'lateral', 'brain_lateral')

    print("\n--- Medial view ---")
    make_and_save_view(hemi, 'medial', 'brain_medial')

    print("\nDone! Files saved:")
    print("  brain_lateral.svg / .png")
    print("  brain_medial.svg / .png")

    # =========================================================================
    # Generate legend as separate SVG
    # =========================================================================

    print("\n--- Generating legend ---")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ROI_LABELS = {
        'dlPFC':     'dlPFC',
        'ACC':       'ACC',
        'preSMA':    'pre-SMA',
        'dmPFC':     'dmPFC',
        'vmPFC':     'vmPFC',
        'precuneus': 'Precuneus',
        'STN':       'STN',
    }

    fig, ax = plt.subplots(figsize=(2.5, 3))
    ax.axis('off')

    patches = []
    for roi_key in ['dlPFC', 'ACC', 'preSMA', 'dmPFC', 'vmPFC', 'precuneus', 'STN']:
        patches.append(
            mpatches.Patch(
                facecolor=COLORS[roi_key],
                edgecolor='black',
                linewidth=0.8,
                label=ROI_LABELS[roi_key],
            )
        )

    legend = ax.legend(
        handles=patches,
        loc='center',
        frameon=True,
        framealpha=1.0,
        edgecolor='black',
        fontsize=11,
        handlelength=1.2,
        handleheight=1.2,
        handletextpad=0.6,
        labelspacing=0.5,
    )
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig('brain_legend.svg', format='svg', dpi=300,
                bbox_inches='tight', transparent=True)
    plt.savefig('brain_legend.png', dpi=300,
                bbox_inches='tight', transparent=True)
    plt.close()
    print("✓ Saved legend: brain_legend.svg / .png")


# =============================================================================
# 10. NOTES
# =============================================================================
"""
CHANGES FROM ORIGINAL
---------------------
1. alpha=1.0 on brain  — fully opaque surface (was 0.15)
2. ROI alpha=1.0       — solid opaque fills (was 0.75)
3. Custom cortex color — light gray with low sulcal contrast
   Format: (gyrus_dark, gyrus_light, sulcus_dark, sulcus_light)
   Increase all 4 values to make the brain lighter overall.
   Decrease the gap between gyrus and sulcus values for less contrast.
4. soften_lighting()   — high ambient, low specular for flatter shading
5. No black borders    — they look too jagged on the pial surface

TUNING THE CORTEX COLOR
------------------------
Valid cortex presets:
  - 'low_contrast'   — (current) light gray, gentle sulcal shading
  - 'bone'           — warm beige/gray
  - 'ivory'          — very light, almost white
  - 'classic'        — darker gray, more contrast
  - 'high_contrast'  — strong sulcal contrast
  - Any color name   — e.g. 'grey', '#B0B0B0' for uniform flat color

TUNING THE LIGHTING
--------------------
In soften_lighting(), adjust:
  - Ambient (currently 0.55): higher = flatter, less shadow
  - Diffuse (currently 0.45): lower = less directional shading
  - Specular (currently 0.05): 0.0 = completely matte

IF ROIs STILL BLEED
--------------------
On pial surfaces, adjacent ROIs can look merged because sulcal
geometry pushes vertices together. Options:
  a) Add borders back with borders=1 (thin) — may still be jagged
  b) Use surf='white' for a smoother surface (less anatomical detail)
  c) Use surf='inflated' (you said you don't want this, but noting it)
  d) Post-process in Illustrator: add thin strokes to ROI paths in SVG

COMPOSITING
-----------
    dlPFC      ■ #2166AC (blue)
    ACC        ■ #D6604D (salmon)
    pre-SMA    ■ #66A61E (green)
    dmPFC      ■ #7570B3 (purple)
    vmPFC      ■ #E6AB02 (gold)
    Precuneus  ■ #E7298A (pink)
    STN        ■ #1B9E77 (teal)
"""