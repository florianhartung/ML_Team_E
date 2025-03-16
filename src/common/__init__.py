NUM_PIXELS = 1039

FEATURES_TRUE_SHOWER = [
    "true_energy",
    "true_theta",
    "true_phi",
    "true_telescope_theta",
    "true_telescope_phi",
    "true_first_interaction_height",
    "true_impact_m1",
    "true_impact_m2",
]

FEATURES_HILLAS = [
    "hillas_length_m1",
    "hillas_width_m1",
    "hillas_delta_m1",
    "hillas_size_m1",
    "hillas_cog_x_m1",
    "hillas_cog_y_m1",
    "hillas_sin_delta_m1",
    "hillas_cos_delta_m1",
    "hillas_length_m2",
    "hillas_width_m2",
    "hillas_delta_m2",
    "hillas_size_m2",
    "hillas_cog_x_m2",
    "hillas_cog_y_m2",
    "hillas_sin_delta_m2",
    "hillas_cos_delta_m2",
]

FEATURES_STEREO = [
    "stereo_direction_x",
    "stereo_direction_y",
    "stereo_zenith",
    "stereo_azimuth",
    "stereo_dec",
    "stereo_ra",
    "stereo_theta2",
    "stereo_core_x",
    "stereo_core_y",
    "stereo_impact_m1",
    "stereo_impact_m2",
    "stereo_impact_azimuth_m1",
    "stereo_impact_azimuth_m2",
    "stereo_shower_max_height",
    "stereo_xmax",
    "stereo_cherenkov_radius",
    "stereo_cherenkov_density",
    "stereo_baseline_phi_m1",
    "stereo_baseline_phi_m2",
    "stereo_image_angle",
    "stereo_cos_between_shower",
]

FEATURES_POINTING = [
    "pointing_zenith",
    "pointing_azimuth",
]

FEATURES_SOURCE = [
    "source_alpha_m1",
    "source_dist_m1",
    "source_cos_delta_alpha_m1",
    "source_dca_m1",
    "source_dca_delta_m1",
    "source_alpha_m2",
    "source_dist_m2",
    "source_cos_delta_alpha_m2",
    "source_dca_m2",
    "source_dca_delta_m2",
]

FEATURES_TIME = [
    "time_gradient_m1",
    "time_gradient_m2",
]

FEATURES_IMAGE_M1 = [f"image_m1_{i}" for i in range(NUM_PIXELS)]
FEATURES_IMAGE_M2 = [f"image_m2_{i}" for i in range(NUM_PIXELS)]
FEATURES_CLEAN_IMAGE_M1 = [f"clean_image_m1_{i}" for i in range(NUM_PIXELS)]
FEATURES_CLEAN_IMAGE_M2 = [f"clean_image_m2_{i}" for i in range(NUM_PIXELS)]
FEATURES_TIMING_M1 = [f"timing_m1_{i}" for i in range(NUM_PIXELS)]
FEATURES_TIMING_M2 = [f"timing_m2_{i}" for i in range(NUM_PIXELS)]
