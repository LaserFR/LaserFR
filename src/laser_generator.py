import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from skimage import exposure
from scipy.ndimage import zoom
import os
import imageio


class LaserModel:
    def __init__(self, P, wavelength, theta, n, d, f, t):
        """
        Initialize the LaserModel class with necessary parameters.

        Parameters:
        P (float): Total power of the laser beam.
        wavelength (float): Wavelength of the laser light.
        theta (float): Divergence angle.
        n (float): Refractive index.
        d (float): Distance between the lens and the detector (CMOS imaging plane).
        f (float): Focal length of the lens.
        t (float): Lens thickness.
        """
        self.P = P
        self.wavelength = wavelength
        self.theta = theta
        self.omega_0 = wavelength / (np.pi * theta)
        self.n = n
        self.d = d
        self.f = f
        self.t = t

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

    def annular_interference_pattern(self, r_a, z):
        """
        Calculate the annular interference pattern.

        Parameters:
        r_a (float): Aperture radius.
        z (float): Distance propagated from the plane where the wavefront is flat.

        Returns:
        np.ndarray: 2D array of interference pattern values.
        """
        P_a = self.integrated_power(r_a, z)
        angle = np.arcsin(self.n * np.sin(np.arctan(self.d / self.f)))
        x = np.linspace(-r_a, r_a, 8000)
        y = np.linspace(-r_a, r_a, 8000)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X ** 2 + Y ** 2)
        P_I = P_a * (np.cos(2 * np.pi * self.n * self.t * np.cos(angle) * R / self.wavelength)) ** 2
        return X, Y, P_I * np.exp(-2 * R ** 2 / self.gaussian_beam_radius(z) ** 2)


class CMOSSensor:
    def __init__(self, width, height, pixel_size, QE_r, QE_g, QE_b, exposure_time, scale_factor):
        """
        Initialize the CMOSSensor class with necessary parameters.

        Parameters:
        width (float): Width of the CMOS sensor in meters.
        height (float): Height of the CMOS sensor in meters.
        pixel_size (float): Size of a single pixel in meters.
        QE_r (float): Quantum Efficiency for the red channel.
        QE_g (float): Quantum Efficiency for the green channel.
        QE_b (float): Quantum Efficiency for the blue channel.
        exposure_time (float): Exposure time in seconds.
        """
        self.width = width
        self.height = height
        self.pixel_size = pixel_size
        self.resolution_x = int(width / pixel_size)
        self.resolution_y = int(height / pixel_size)
        self.QE_r = QE_r
        self.QE_g = QE_g
        self.QE_b = QE_b
        self.exposure_time = exposure_time
        self.scale_factor = scale_factor

    def power_to_photons(self, power, wavelength, QE):
        """
        Convert laser power to photon counts based on the quantum efficiency.

        Parameters:
        power (np.ndarray): Power distribution array.
        wavelength (float): Wavelength of the laser light.
        QE (float): Quantum Efficiency for the given channel.

        Returns:
        np.ndarray: Array of photon counts.
        """
        h = 6.62607015e-34  # Planck's constant (J*s)
        c = 3e8  # Speed of light (m/s)
        energy_per_photon = h * c / wavelength
        photon_count = (power * QE * self.exposure_time) / energy_per_photon
        return photon_count

    def integrate_photons(self, photon_counts, high_res_x, high_res_y):
        """
        Integrate photon counts over the pixel grid of the sensor.

        Parameters:
        photon_counts (np.ndarray): High-resolution array of photon counts.
        high_res_x (int): Number of columns in the high-resolution grid.
        high_res_y (int): Number of rows in the high-resolution grid.

        Returns:
        np.ndarray: Integrated photon counts over the sensor's pixel grid.
        """
        resampled_laser_power_density = zoom(photon_counts, (
        self.resolution_y / high_res_y, self.resolution_x / high_res_x))

        # integrated_photons = np.zeros((self.resolution_y, self.resolution_x))
        # factor_x = high_res_x // self.resolution_x
        # factor_y = high_res_y // self.resolution_y
        # if factor_x <= 0 or factor_y <= 0:
        #     raise ValueError("High-resolution dimensions must be greater than the sensor's resolution dimensions.")
        #
        # for i in range(self.resolution_y):
        #     for j in range(self.resolution_x):
        #         start_y = i * factor_y
        #         end_y = start_y + factor_y
        #         start_x = j * factor_x
        #         end_x = start_x + factor_x
        #         if end_y > high_res_y:  # Handle boundary conditions
        #             end_y = high_res_y
        #         if end_x > high_res_x:  # Handle boundary conditions
        #             end_x = high_res_x
        #         subarray = photon_counts[start_y:end_y, start_x:end_x]
        #         integrated_photons[i, j] = np.sum(subarray)
        #
        # # for i in range(self.resolution_y):
        # #     for j in range(self.resolution_x):
        # #         subarray = photon_counts[i * factor_y:(i + 1) * factor_y, j * factor_x:(j + 1) * factor_x]
        # #         integrated_photons[i, j] = np.sum(subarray)
        # # plt.figure(figsize=(16, 16))
        # # plt.imshow(integrated_photons, origin='lower')
        # return integrated_photons
        return resampled_laser_power_density

    def normalize_to_rgb(self, photons_r, photons_g, photons_b):
        """
        Integrate photon counts over the pixel grid of the sensor.

        Parameters:
        photon_counts (np.ndarray): High-resolution array of photon counts.
        high_res_x (int): Number of columns in the high-resolution grid.
        high_res_y (int): Number of rows in the high-resolution grid.

        Returns:
        np.ndarray: Integrated photon counts over the sensor's pixel grid.
        """
        # Avoid division by zero
        max_photons_r = np.max(photons_r) if np.max(photons_r) > 0 else 1
        max_photons_g = np.max(photons_g) if np.max(photons_g) > 0 else 1
        max_photons_b = np.max(photons_b) if np.max(photons_b) > 0 else 1
        max_photons = max(max_photons_r, max_photons_g, max_photons_b)

        norm_r = (photons_r / max_photons) * self.scale_factor
        norm_g = (photons_g / max_photons) * self.scale_factor
        norm_b = (photons_b / max_photons) * self.scale_factor

        # # Apply gamma correction
        # norm_r = np.power(norm_r, 1 / gamma)
        # norm_g = np.power(norm_g, 1 / gamma)
        # norm_b = np.power(norm_b, 1 / gamma)

        # Scale to 0-255 range
        norm_r = (norm_r * 255).astype(np.uint8)
        norm_g = (norm_g * 255).astype(np.uint8)
        norm_b = (norm_b * 255).astype(np.uint8)

        rgb_image = np.stack((norm_r, norm_g, norm_b), axis=-1)
        return rgb_image


if __name__ == '__main__':
    P = 200.0  # Example power in mW
    wavelength = 0.000000785  # Example wavelength in meters (785 nm)
    theta = 0.028  # Example divergence angle in radians
    n = 1.5  # Example refractive index
    d = 0.004  # Example distance from aperture to CMOS imaging plane center in meters
    f = 0.013  # Example focal length in meters
    z = 0.35  # Example distance from the beam waist in meters
    t = 0.003  # Example lens thickness in meters
    r_a = 0.0072  # Example aperture radius in meters

    laser_model = LaserModel(P, wavelength, theta, n, d, f, t)
    X, Y, P_I = laser_model.annular_interference_pattern(r_a, z)

    # Sensor parameters
    sensor_width = 0.003024  # Sensor width in meters
    sensor_height = 0.003024  # Sensor height in meters
    pixel_size = 1e-6  # Pixel size in meters (1.0µm)
    QE_r = 0.33  # Quantum Efficiency for red
    QE_g = 0.14  # Quantum Efficiency for green
    QE_b = 0.23  # Quantum Efficiency for blue
    exposure_time = 1/30  # Example exposure time in seconds
    scale_factor = P / 200

    sensor = CMOSSensor(sensor_width, sensor_height, pixel_size, QE_r, QE_g, QE_b, exposure_time, scale_factor)

    # Convert power to photon counts for each color channel
    photons_r = sensor.power_to_photons(P_I, wavelength, QE_r)
    photons_g = sensor.power_to_photons(P_I, wavelength, QE_g)
    photons_b = sensor.power_to_photons(P_I, wavelength, QE_b)

    high_res_x, high_res_y = X.shape

    # Integrate photons over pixels for each color channel
    integrated_photons_r = sensor.integrate_photons(photons_r, high_res_x, high_res_y)
    integrated_photons_g = sensor.integrate_photons(photons_g, high_res_x, high_res_y)
    integrated_photons_b = sensor.integrate_photons(photons_b, high_res_x, high_res_y)

    # Normalize to RGB values
    rgb_values = sensor.normalize_to_rgb(integrated_photons_r, integrated_photons_g, integrated_photons_b)

    # Save the RGB image
    output_dir = "../data/laser_images"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"laser_{int(P)}.png")
    imageio.imwrite(filename, rgb_values)

    # # Display the RGB image
    # plt.figure(figsize=(16, 16))
    # plt.imshow(rgb_values, origin='lower')
    # plt.colorbar(label='Intensity')
    # plt.xlabel('X (pixels)')
    # plt.ylabel('Y (pixels)')
    # plt.title('Laser Interference Pattern as RGB Values')
    # plt.show()
