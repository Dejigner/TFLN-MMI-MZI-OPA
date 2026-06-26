# phase1_mode_sweep.py
# MPB: neff vs. waveguide width for 300 nm X-cut TFLN strip at 1550 nm
# Reference geometry: [R1], [R4], [R5]

#notes to self, MPB was first created for another intended purpose than integrated photonics, so slight change in terminilogy or variables
# k here is beta (wavevector)
# Neff in lec --> beta / k0
#Bands here refer to modes
# MPB defines things with cycles instead of radians so beta = 2*pi*kz
#plane waves deifinition here is ~ e^(j2*pi*Kz*z) so kz = 1/lambda guide
# we studied ~ e^(j beta*z) and beta = (2 pi)/lambda guide

from meep import mpb
import meep as mp
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# ── Units: µm throughout; c = 1 ────────────────────────────────────────────
lam = 1.55          # µm
f0  = 1.0 / lam    # = 0.6452 in MEEP units (a = 1 µm, c = 1)

# ── Material indices at 1550 nm (Zelmon et al. 1997 [R5]) ──────────────────
n_e  = 2.138   # LNO extraordinary (TE mode, X-cut: c-axis along x)
n_o  = 2.211   # LNO ordinary
n_ox = 1.444   # SiO₂ BOX
n_air = 1.0

def lno_medium():
    """X-cut LNO: extraordinary along x → epsilon_xx = ne^2, yy = zz = no^2"""
    return mp.Medium(epsilon_diag=mp.Vector3(n_e**2, n_o**2, n_o**2))

# ── Supercell (cross-section in x-y; z = propagation direction) ────────────
sx = 6.0   # x-span (µm) — must be >> wg width for PML-like confinement
sy = 5.5   # y-span (µm) — must contain BOX + LNO + air above
h  = 0.30  # LNO thickness

# BOX sits below LNO center:  y_center_BOX = -(h/2 + BOX_thickness/2)
# Air above LNO: default background


def run_mpb(w, resolution=64, verbose=False):
    """
    Solve for guided modes of a strip waveguide of width w (µm).
    Returns list of neff values at 1550 nm.
    """
    geometry = [
        # SiO₂ BOX — placed below LNO layer center (y=0 = LNO center)
        mp.Block(
            size=mp.Vector3(sx, 2.5, mp.inf),
            center=mp.Vector3(0, -(h/2 + 1.25), 0),
            material=mp.Medium(index=n_ox)
        ),
        # LNO waveguide core
        mp.Block(
            size=mp.Vector3(w, h, mp.inf),
            center=mp.Vector3(0, 0, 0),
            material=lno_medium()
        ),
    ]
    # Default background = air (n=1)

    ms = mpb.ModeSolver(
        geometry_lattice=mp.Lattice(size=mp.Vector3(sx, sy, 0)),
        geometry=geometry,
        resolution=resolution,  # 64 pts/µm → ~16 nm sampling in core
        num_bands=4,            # look for up to 4 guided modes
        default_material=mp.Medium(index=n_ox),
    )

    if not verbose:
        ms.filename_prefix = 'tmp'

    # find_k: at fixed frequency f0, find k_z for each guided band
    # Lower bound: k_z > n_ox * f0 (mode must be above BOX cutoff)
    # Upper bound: k_z < n_e  * f0 (mode must be below core index)
    neff_vals = []
    
    for mode_n in range (1,5):
        try:
            k_pts = ms.find_k(
                mp.NO_PARITY,           # allow all parities
                f0,                     # target frequency (in MEEP units)
                mode_n, mode_n,                   # band range: 1 to 4
                mp.Vector3(0, 0, 1),    # propagation along z
                1e-4,                   # convergence tolerance
                kmag_guess=f0 * ((n_e + n_ox) / 2), # **NEW: Initial guess**
                kmag_min=f0 * n_ox,       # k lower bound (BOX cutoff)
                kmag_max=f0 * n_e,        # k upper bound (Core index)
                # f0 * n_ox,              # k lower bound
                # f0 * n_e,               # k upper bound
            )

            if k_pts and k_pts[0] > f0 * n_ox:
                neff_vals.append(k_pts[0] / f0)

            # # neff = k_z / f0  (since λ = 1/f0 in units where c = 1)
            # # neff_vals = []
            # for k in k_pts:
            #     if k.z > f0 * n_ox:  # only guided modes (above BOX)
            #         neff_vals.append(k.z / f0)

            # return neff_vals

        except ValueError:
            break

    return neff_vals


def compute_ng(w, delta_lam=0.01):
    """
    Group index ng = neff - λ × d(neff)/dλ
    Estimated via central finite difference over ±delta_lam around 1550 nm.
    """
    global f0, lam

    lam_lo = lam - delta_lam
    lam_hi = lam + delta_lam

    # Temporarily change f0 for each wavelength
    f0_lo = 1.0 / lam_lo
    f0_hi = 1.0 / lam_hi
    f0_0  = 1.0 / lam

    def _get_neff_at_f(f):
        """Run MPB at a different frequency."""
        # (In practice, recreate ModeSolver with new f; shortened here)
        ms = mpb.ModeSolver(
            geometry_lattice=mp.Lattice(size=mp.Vector3(sx, sy, 0)),
            geometry=[
                mp.Block(mp.Vector3(sx, 2.5, mp.inf),
                         center=mp.Vector3(0, -(h/2 + 1.25), 0),
                         material=mp.Medium(index=n_ox)),
                mp.Block(mp.Vector3(w, h, mp.inf),
                         center=mp.Vector3(0, 0, 0),
                         material=lno_medium()),
            ],
            resolution=64, num_bands=2,
            default_material=mp.Medium(index=n_ox),
        )
        k_pts = ms.find_k(mp.NO_PARITY, f, 1, 2, mp.Vector3(0,0,1),
                          1e-4, 
                        #   f*n_ox, f*n_e,
                          kmag_guess=f * ((n_e + n_ox) / 2), # **NEW: Initial guess**
                          kmag_min=f * n_ox,       # k lower bound (BOX cutoff)
                          kmag_max=f * n_e, #
                          
                          )
        if k_pts and k_pts[0] > f * n_ox:
            return k_pts[0] / f
        return None

    neff_lo = _get_neff_at_f(f0_lo)
    neff_hi = _get_neff_at_f(f0_hi)
    neff_0  = _get_neff_at_f(f0_0)

    if None in (neff_lo, neff_hi, neff_0):
        return None, None

    dneff_dlam = (neff_hi - neff_lo) / (2 * delta_lam)
    ng = neff_0 - lam * dneff_dlam
    return ng, neff_0


# ── Width sweep ─────────────────────────────────────────────────────────────
widths = np.arange(0.6, 1.45, 0.1)   # 0.6 to 1.4 µm in 0.1 steps
results = []

if mp.am_master():
    print(f"{'w (µm)':>8}  {'neff_1':>8}  {'neff_2':>8}  {'# modes':>8}")
for w in widths:
    neff_list = run_mpb(w, resolution=64, verbose=False)
    n1 = neff_list[0] if len(neff_list) > 0 else None
    n2 = neff_list[1] if len(neff_list) > 1 else None
    n_modes = len(neff_list)
    results.append({'w': w, 'neff1': n1, 'neff2': n2, 'n_modes': n_modes})
    if mp.am_master():
        print(f"{w:>8.2f}  {str(round(n1,4)) if n1 else 'none':>8}  "
          f"{str(round(n2,4)) if n2 else 'none':>8}  {n_modes:>8}")

# ── Group index at design width ─────────────────────────────────────────────
ng, neff_design = compute_ng(0.90)

# if mp.am_master():
#     print(f"\nAt w = 0.90 µm: neff = {neff_design:.4f}, ng = {ng:.4f}")

if mp.am_master():
    if ng is not None and neff_design is not None:
        print(f"\nAt w = 0.90 µm: neff = {neff_design:.4f}, ng = {ng:.4f}")
    else:
        print("\nAt w = 0.90 µm: No guided mode found.")

# ── Save for INTERCONNECT and Lumerical ─────────────────────────────────────
if mp.am_master():
    w_arr    = np.array([r['w']    for r in results])
    neff_arr = np.array([r['neff1'] for r in results], dtype=float)

    scipy.io.savemat('neff_vs_width.mat', {
        'widths': w_arr,
        'neff':   neff_arr,
        'ng_design': ng,
        'neff_design': neff_design,
    })
    print("Saved: neff_vs_width.mat")

    # ── Plot ─────────────────────────────────────────────────────────────────────
    plt.figure(figsize=(7, 4))
    plt.plot(w_arr, neff_arr, 'b-o', label='neff TE00')
    plt.axvline(0.9, color='gray', linestyle='--', label='w = 0.9 µm')
    plt.xlabel('Waveguide width (µm)')
    plt.ylabel('Effective index neff')
    plt.title('neff vs. Width — 300 nm X-cut TFLN Strip (MPB)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('neff_vs_width.png', dpi=150)
    plt.close()