import numpy as np
from scipy.integrate import quad


class LaserModel:
    def __init__(self, P, wavelength, omega_0, n, d, f):
        """
        Initialize the LaserModel class with necessary parameters.

        Parameters:
        P (float): Total power of the laser beam.
        wavelength (float): Wavelength of the laser light.
        omega_0 (float): Beam waist (the minimum beam radius).
        n (float): Refractive index.
        d (float): Distance from the aperture to the lens.
        f (float): Focal length of the lens.
        """
        self.P = P
        self.wavelength = wavelength
        self.omega_0 = omega_0
        self.theta = wavelength / (np.pi * omega_0)
        self.n = n
        self.d = d
        self.f = f

    def gaussian_beam_radius(self, z):
        """
        Calculate the beam radius at a distance z from the beam waist.

        Parameters:
        z (float): Distance propagated from the plane where the wavefront is flat.

        Returns:
        float: Beam radius at distance z.
        """
        omega_z = self.omega_0 * np.sqrt(1 + (self.wavelength * z / (np.pi * self.omega_0 ** 2)) ** 2)
        return omega_z

    def gaussian_beam_profile(self, r, z):
        """
        Calculate the Gaussian beam profile.

        Parameters:
        r (float): Radial distance from the beam center.
        z (float): Distance propagated from the plane where the wavefront is flat.

        Returns:
        float: Irradiance at distance r.
        """
        omega_z = self.gaussian_beam_radius(z)
        P_r = (2 * self.P / (np.pi * omega_z ** 2)) * np.exp(-2 * r ** 2 / omega_z ** 2)
        return P_r

    def power_intensity(self, r, z):
        """
        Calculate the power intensity at a location (z, r).

        Parameters:
        r (float): Radial distance from the beam center.
        z (float): Distance propagated from the plane where the wavefront is flat.

        Returns:
        float: Power intensity at the location (z, r).
        """
        numerator = self.wavelength ** 2 + np.pi * z ** 2 * self.theta ** 4
        denominator = np.pi * self.theta ** 2
        exponent = -2 * np.pi ** 2 * r ** 2 * self.theta ** 2 / (
                    self.wavelength ** 2 + np.pi ** 2 * z ** 2 * self.theta ** 2)
        P_r = (numerator / denominator) * np.exp(exponent)
        return P_r

    def integrated_power(self, r_a, z):
        """
        Calculate the integrated power within the aperture radius.

        Parameters:
        r_a (float): Aperture radius.
        z (float): Distance propagated from the plane where the wavefront is flat.

        Returns:
        float: Integrated power P_a.
        """
        integrand = lambda r: self.gaussian_beam_profile(r, z) * r
        P_a, _ = quad(integrand, 0, r_a)
        return P_a * 2 * np.pi  # Since the integration is in polar coordinates

    def annular_interference_pattern(self, r_a, z, t):
        """
        Calculate the annular interference pattern.

        Parameters:
        r_a (float): Aperture radius.
        z (float): Distance propagated from the plane where the wavefront is flat.
        t (float): Time.

        Returns:
        float: Interference pattern value at time t.
        """
        P_a = self.integrated_power(r_a, z)
        angle = np.arcsin(self.n * np.sin(np.arctan(self.d / self.f)))
        P_I = P_a * np.cos(2 * np.pi * self.n * t * np.cos(angle) / self.wavelength) ** 2
        return P_I


if __name__ == '__main__':
    P = 5.0  # Example power in mW
    wavelength = 0.000000785  # Example wavelength in meters (785 nm)
    omega_0 = 0.001  # Example beam waist in meters
    n = 1.0  # Example refractive index
    d = 0.1  # Example distance from aperture to lens in meters
    f = 0.15  # Example focal length in meters
    r = 0.01  # Example radial distance in meters
    z = 0.35  # Example distance from the beam waist in meters
    t = 1.0  # Example time in seconds
    r_a = 0.005  # Example aperture radius in meters

    laser_model = LaserModel(P, wavelength, omega_0, n, d, f)
    omega_z = laser_model.gaussian_beam_radius(z)
    P_r = laser_model.gaussian_beam_profile(r, z)
    P_intensity = laser_model.power_intensity(r, z)
    P_I = laser_model.annular_interference_pattern(r_a, z, t)

    print(f"Beam Radius at z={z} m: {omega_z}")
    print(f"Gaussian Beam Profile at (r={r} m, z={z} m): {P_r}")
    print(f"Power Intensity at (r={r} m, z={z} m): {P_intensity}")
    print(f"Annular Interference Pattern at t={t} s: {P_I}")
