"""Temperature-dependent continuum channeling potentials for elemental and polyatomic crystals.

Modules
  form_factors         Doyle-Turner/Peng (1996) and Moliere form factors in q space
  thermal              u(T) / U_ij(T): tiers A (Peng table), B (Debye), C (Trots, Einstein, ADP hook)
  structure            CIF loading, optional thermal expansion, materials table, axis frames
  potential_fourier    reciprocal-space axial and planar potentials (default)
  potential_realspace  Gaussian-sum strings and planes (reference method, Peng form factors only)
  observables          well depth, maximum field, Lindhard angle, U(x_c)
"""
