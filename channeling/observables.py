"""Quantities derived from the potentials."""
import numpy as np

EV_PER_ANG_TO_GEV_PER_CM = 0.1   # 1 eV/A = 1e8 eV/cm = 0.1 GeV/cm


def lindhard_angle_urad(U0_eV, energy_GeV):
    """Lindhard critical angle sqrt(2 U0/(p v)) [urad], ultrarelativistic (p v = E)."""
    return float(np.sqrt(2 * U0_eV / (energy_GeV * 1e9)) * 1e6)


def field_GeV_per_cm(e_eV_per_A):
    return float(e_eV_per_A * EV_PER_ANG_TO_GEV_PER_CM)


def planar_U_at_xc(x, U, d, u):
    """Planar potential U(x_c) [eV] relative to the channel centre at x_c = D - 2u, with D the
    distance from the channel minimum to the nearest plane (maximum of U); u [A] is the thermal
    amplitude of the planes. Biryukov et al. (Table 1.3) define the planar critical potential so.
    Returns (U(x_c), x_c, dU/dx at x_c in GeV/cm)."""
    one = (x >= 0) & (x < d)
    x0 = x[one][np.argmin(U[one])]
    # nearest maximum to the right of the minimum, within one period
    right = (x > x0) & (x <= x0 + d)
    xp = x[right][np.argmax(U[right])]
    D = xp - x0
    xc = D - 2 * u
    U0 = np.interp(x0, x, U)
    Uxc = np.interp(x0 + xc, x, U) - U0
    h = float(x[1] - x[0])
    F = (np.interp(x0 + xc + h, x, U) - np.interp(x0 + xc - h, x, U)) / (2 * h)
    return float(Uxc), float(xc), float(F * EV_PER_ANG_TO_GEV_PER_CM)
