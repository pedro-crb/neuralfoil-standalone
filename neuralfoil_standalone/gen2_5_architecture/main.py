try:
    import optisandbox.numpy as np
    from optisandbox.numpy import length
    min_fn = np.softmin
    max_fn = np.softmax
except ImportError:
    import numpy as np
    length = len

    # noinspection PyUnusedLocal
    def min_fn(*args, softness=None, hardness=None):
        return np.min(args, axis=0)

    # noinspection PyUnusedLocal
    def max_fn(*args, softness=None, hardness=None):
        return np.max(args, axis=0)

from typing import Union, Dict, Set, List
from pathlib import Path

npz_file_directory = Path(__file__).parent / "nn_weights_and_biases"

NUM_BL_POINTS = 32


def compute_optimal_x_points(n_points):
    s = np.linspace(0, 1, n_points + 1)
    return (s[1:] + s[:-1]) / 2


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _inverse_sigmoid(x):
    return -np.log(1 / x - 1)


_scaled_input_distribution = dict(np.load(npz_file_directory / "scaled_input_distribution.npz"))
_scaled_input_distribution["N_inputs"] = len(_scaled_input_distribution["mean_inputs_scaled"])


def _squared_mahalanobis_distance(x):
    d = _scaled_input_distribution
    mean = np.reshape(d["mean_inputs_scaled"], (1, -1))
    x_minus_mean = (x.T - mean.T).T
    return np.sum(
        x_minus_mean @ d["inv_cov_inputs_scaled"] * x_minus_mean,
        axis=1
    )


def _blend(switch: float, value_switch_high, value_switch_low):
    weight_to_value_switch_high = 0.5*np.tanh(switch) + 0.5
    blend_value = (
            value_switch_high * weight_to_value_switch_high +
            value_switch_low * (1 - weight_to_value_switch_high)
    )
    return blend_value


def _cosine_hermite_patch(
        x: Union[float, np.ndarray], x_a: float, x_b: float, f_a: float, f_b: float,
        dfdx_a: float, dfdx_b: float
) -> Union[float, np.ndarray]:
    t = (x - x_a) / (x_b - x_a)
    l1 = (x - x_a) * dfdx_a + f_a
    l2 = (x - x_b) * dfdx_b + f_b
    b = 0.5 + 0.5 * np.cos(np.pi * t)
    return b * l1 + (1 - b) * l2


def _post_stall_model(alpha: float):
    sina = np.sin(np.radians(alpha))
    cosa = np.cos(np.radians(alpha))

    Cd90_0 = 2.08
    pn2_star = 8.36e-2
    pn3_star = 4.06e-1
    pt1_star = 9.00e-2
    pt2_star = -1.78e-1
    pt3_star = -2.98e-1

    Cd90 = Cd90_0 + pn2_star * cosa + pn3_star * cosa ** 2
    CN = Cd90 * sina

    CT = (pt1_star + pt2_star * cosa + pt3_star * cosa ** 3) * sina ** 2

    CL = CN * cosa + CT * sina
    CD = CN * sina - CT * cosa

    # Crude fit from:
    # "Ma Z, Smeur EJJ, de Croon GCHE. Wind tunnel tests of a wing at all angles of attack.
    # International Journal of Micro Air Vehicles. 2022;14."
    # with 3D to 2D correction
    CM = np.zeros_like(alpha)

    return CL, CD, CM


def get_aero_from_kulfan_parameters(
        kulfan_parameters: Dict[str, Union[float, np.ndarray]],
        alpha: Union[float, np.ndarray],
        Re: Union[float, np.ndarray],
        n_crit: Union[float, np.ndarray] = 9.0,
        xtr_upper: Union[float, np.ndarray] = 1.0,
        xtr_lower: Union[float, np.ndarray] = 1.0,
        model_size="large"
) -> Dict[str, Union[float, np.ndarray]]:
    alpha = np.array(alpha, dtype='float')
    Re = np.array(Re, dtype='float')
    n_crit = np.array(n_crit, dtype='float')
    xtr_upper = np.array(xtr_upper, dtype='float')
    xtr_lower = np.array(xtr_lower, dtype='float')

    filename = npz_file_directory / f"nn-{model_size}.npz"
    if not filename.exists():
        raise FileNotFoundError(
            f"Could not find the neural network file {filename}, which contains the weights and biases.")

    data: Dict[str, np.ndarray] = np.load(filename)

    input_rows: List[Union[float, np.ndarray]] = [
        *[kulfan_parameters["upper_weights"][i] for i in range(8)],
        *[kulfan_parameters["lower_weights"][i] for i in range(8)],
        kulfan_parameters["leading_edge_weight"],
        kulfan_parameters["TE_thickness"] * 50,
        np.sin(np.radians(2 * alpha)),
        np.cos(np.radians(alpha)),
        1 - np.cos(np.radians(alpha)) ** 2,
        (np.log(Re) - 12.5) / 3.5,
        # No mach
        (n_crit - 9) / 4.5,
        xtr_upper,
        xtr_lower,
    ]
    N_cases = 1
    for row in input_rows:
        if length(np.atleast_1d(row)) > 1:
            if N_cases == 1:
                N_cases = length(row)
            else:
                if length(row) != N_cases:
                    raise ValueError(
                        f"The inputs to the neural network must all have the same length. "
                        f"(Conflicting lengths: {N_cases} and {length(row)})"
                    )

    for i, row in enumerate(input_rows):
        input_rows[i] = np.ones(N_cases) * row

    x = np.stack(input_rows, axis=1)  # N_cases x N_inputs
    # Evaluate the neural network
    # First, determine what the structure of the neural network is (i.e., how many layers it has) by looking at the keys
    # These keys come from the dictionary of saved weights/biases for the specified neural network.
    try:
        layer_indices: Set[int] = set([
            int(key.split(".")[1])
            for key in data.keys()
        ])
    except TypeError:
        raise ValueError(
            f"Got an unexpected neural network file format.\n"
            f"Dictionary keys should be strings of the form 'net.0.weight', 'net.0.bias', 'net.2.weight', etc.'.\n"
            f"Instead, got keys of the form {data.keys()}.\n"
        )
    layer_indices: List[int] = sorted(list(layer_indices))

    # Now, set up evaluation of the basic neural network.
    def net(_x: np.ndarray):
        """
        Evaluates the raw network (taking in scaled inputs and returning scaled outputs).
        Input `x` dims: N_cases x N_inputs
        Output `y` dims: N_cases x N_outputs
        """
        _x = np.transpose(_x)
        layer_indices_to_iterate = layer_indices.copy()

        while len(layer_indices_to_iterate) != 0:
            i = layer_indices_to_iterate.pop(0)
            w = data[f"net.{i}.weight"]
            b = data[f"net.{i}.bias"]
            _x = w @ _x + np.reshape(b, (-1, 1))

            if len(layer_indices_to_iterate) != 0:  # Don't apply the activation function on the last layer
                _x = _x / (1 + np.exp(-_x))
        _x = np.transpose(_x)
        return _x

    y = net(x)  # N_outputs x N_cases
    y[:, 0] = y[:, 0] - _squared_mahalanobis_distance(x) / (2 * _scaled_input_distribution["N_inputs"])
    # This was baked into training in order to ensure the network
    # asymptotes to zero analysis confidence far away from the training data.

    # Then, flip the inputs and evaluate the network again.
    # The goal here is to embed the invariant of "symmetry across alpha" into the network evaluation.
    # (This was also performed during training, so the network is "intended" to be evaluated this way.)

    x_flipped = x + 0.  # This is a array-api-agnostic way to force a memory copy of the array to be made.
    x_flipped[:, :8] = x[:, 8:16] * -1  # switch kulfan_lower with a flipped kulfan_upper
    x_flipped[:, 8:16] = x[:, :8] * -1  # switch kulfan_upper with a flipped kulfan_lower
    x_flipped[:, 16] = -1 * x[:, 16]  # flip kulfan_LE_weight
    x_flipped[:, 18] = -1 * x[:, 18]  # flip sin(2a)
    x_flipped[:, 23] = x[:, 24]  # flip xtr_upper with xtr_lower
    x_flipped[:, 24] = x[:, 23]  # flip xtr_lower with xtr_upper

    y_flipped = net(x_flipped)
    y_flipped[:, 0] = y_flipped[:, 0] - _squared_mahalanobis_distance(x_flipped) / (
            2 * _scaled_input_distribution["N_inputs"]
    )
    # This was baked into training in order to ensure the network
    # asymptotes to zero analysis confidence far away from the training data.
    # The resulting outputs will also be flipped, so we need to flip them back to their normal orientation
    y_unflipped = y_flipped + 0.  # This is a array-api-agnostic way to force a memory copy of the array to be made.
    y_unflipped[:, 1] = y_flipped[:, 1] * -1  # CL
    y_unflipped[:, 3] = y_flipped[:, 3] * -1  # CM
    y_unflipped[:, 4] = y_flipped[:, 5]  # switch Top_Xtr with Bot_Xtr
    y_unflipped[:, 5] = y_flipped[:, 4]  # switch Bot_Xtr with Top_Xtr

    # switch upper and lower Ret, H
    y_unflipped[:, 6:6 + 32 * 2] = y_flipped[:, 6 + 32 * 3: 6 + 32 * 5]
    y_unflipped[:, 6 + 32 * 3: 6 + 32 * 5] = y_flipped[:, 6:6 + 32 * 2]

    # switch upper_bl_ue/vinf with lower_bl_ue/vinf
    y_unflipped[:, 6 + 32 * 2: 6 + 32 * 3] = -1 * y_flipped[:, 6 + 32 * 5: 6 + 32 * 6]
    y_unflipped[:, 6 + 32 * 5: 6 + 32 * 6] = -1 * y_flipped[:, 6 + 32 * 2: 6 + 32 * 3]

    # Then, average the two outputs to get the "symmetric" result
    y_fused = (y + y_unflipped) / 2
    y_fused[:, 0] = _sigmoid(y_fused[:, 0])  # Analysis confidence, a binary variable
    y_fused[:, 4] = np.clip(y_fused[:, 4], 0, 1)  # Top_Xtr
    y_fused[:, 5] = np.clip(y_fused[:, 5], 0, 1)  # Bot_Xtr

    # Unpack outputs
    analysis_confidence = y_fused[:, 0]
    CL = y_fused[:, 1] / 2
    CD = np.exp((y_fused[:, 2] - 2) * 2)
    CM = y_fused[:, 3] / 20
    Top_Xtr = y_fused[:, 4]
    Bot_Xtr = y_fused[:, 5]

    upper_bl_ue_over_vinf = y_fused[:, 6 + NUM_BL_POINTS * 2:6 + NUM_BL_POINTS * 3]
    lower_bl_ue_over_vinf = y_fused[:, 6 + NUM_BL_POINTS * 5:6 + NUM_BL_POINTS * 6]

    upper_theta = (
                          (10 ** y_fused[:, 6: 6 + NUM_BL_POINTS]) - 0.1
                  ) / (np.abs(upper_bl_ue_over_vinf) * np.reshape(Re, (-1, 1)))
    upper_H = 2.6 * np.exp(y_fused[:, 6 + NUM_BL_POINTS: 6 + NUM_BL_POINTS * 2])

    lower_theta = (
                          (10 ** y_fused[:, 6 + NUM_BL_POINTS * 3: 6 + NUM_BL_POINTS * 4]) - 0.1
                  ) / (np.abs(lower_bl_ue_over_vinf) * np.reshape(Re, (-1, 1)))
    lower_H = 2.6 * np.exp(y_fused[:, 6 + NUM_BL_POINTS * 4: 6 + NUM_BL_POINTS * 5])

    results = {
        "analysis_confidence": analysis_confidence,
        "CL": CL,
        "CD": CD,
        "CM": CM,
        "Top_Xtr": Top_Xtr,
        "Bot_Xtr": Bot_Xtr,
        **{
            f"upper_bl_theta_{i}": upper_theta[:, i]
            for i in range(NUM_BL_POINTS)
        },
        **{
            f"upper_bl_H_{i}": upper_H[:, i]
            for i in range(NUM_BL_POINTS)
        },
        **{
            f"upper_bl_ue/vinf_{i}": upper_bl_ue_over_vinf[:, i]
            for i in range(NUM_BL_POINTS)
        },
        **{
            f"lower_bl_theta_{i}": lower_theta[:, i]
            for i in range(NUM_BL_POINTS)
        },
        **{
            f"lower_bl_H_{i}": lower_H[:, i]
            for i in range(NUM_BL_POINTS)
        },
        **{
            f"lower_bl_ue/vinf_{i}": lower_bl_ue_over_vinf[:, i]
            for i in range(NUM_BL_POINTS)
        },
    }
    return {key: np.reshape(value, -1) for key, value in results.items()}


def get_aero_with_corrections(
        kulfan_parameters: Dict[str, Union[float, np.ndarray]],
        alpha: Union[float, np.ndarray],
        Re: Union[float, np.ndarray],
        mach: Union[float, np.ndarray] = 0.,
        n_crit: Union[float, np.ndarray] = 9.0,
        xtr_upper: Union[float, np.ndarray] = 1.0,
        xtr_lower: Union[float, np.ndarray] = 1.0,
        model_size: str = "large",
        control_surface_deflection: Union[float, np.ndarray] = 0.0,
        control_surface_hinge_point: Union[float, np.ndarray] = 0.0,
        wave_drag_foil_thickness: float = 0.12,
        ) -> Dict[str, Union[float, np.ndarray]]:
    # setup:
    alpha = np.array(alpha, dtype='float')
    Re = np.array(Re, dtype='float')
    mach = np.array(mach, dtype='float')
    n_crit = np.array(n_crit, dtype='float')
    xtr_upper = np.array(xtr_upper, dtype='float')
    xtr_lower = np.array(xtr_lower, dtype='float')
    control_surface_deflection = np.array(control_surface_deflection, dtype='float')
    control_surface_hinge_point = np.array(control_surface_hinge_point, dtype='float')

    # Neuralfoil Run
    alpha = np.mod(alpha + 180, 360) - 180
    effectiveness = 1 - np.maximum(0, control_surface_hinge_point + 1e-16) ** 2.751428551177291
    effective_d_alpha = control_surface_deflection * effectiveness
    effective_CD_multiplier = (
            2 + (control_surface_deflection / 11.5) ** 2 - (1 + (control_surface_hinge_point / 11.5) ** 2) ** 0.5
    )
    nf_aero = get_aero_from_kulfan_parameters(
        kulfan_parameters=kulfan_parameters,
        alpha=alpha + effective_d_alpha,
        Re=Re,
        n_crit=n_crit,
        xtr_upper=xtr_upper,
        xtr_lower=xtr_lower,
        model_size=model_size
    )
    CL = nf_aero["CL"]
    CD = nf_aero["CD"] * effective_CD_multiplier
    CM = nf_aero["CM"]
    Cpmin_0 = min_fn(
        *[1 - nf_aero[f"upper_bl_ue/vinf_{i}"] ** 2 for i in range(NUM_BL_POINTS)],
        *[1 - nf_aero[f"lower_bl_ue/vinf_{i}"] ** 2 for i in range(NUM_BL_POINTS)],
        softness=0.01
    )
    Top_Xtr = nf_aero["Top_Xtr"]
    Bot_Xtr = nf_aero["Bot_Xtr"]

    # 360 degrees correction
    CL_if_separated, CD_if_separated, CM_if_separated = _post_stall_model(alpha)
    alpha_stall_positive = 20
    alpha_stall_negative = -20
    is_separated = max_fn(alpha - alpha_stall_positive, alpha_stall_negative - alpha) / 3
    CL = _blend(is_separated, CL_if_separated, CL)
    CD = np.exp(_blend(is_separated, np.log(CD_if_separated + 0.074/Re**0.2), np.log(CD)))
    CM = _blend(is_separated, CM_if_separated, CM)
    Top_Xtr = _blend(is_separated, 0.5 - 0.5 * np.tanh(10 * np.sin(np.radians(alpha))), Top_Xtr)
    Bot_Xtr = _blend(is_separated, 0.5 + 0.5 * np.tanh(10 * np.sin(np.radians(alpha))), Bot_Xtr)

    # Compressibility base corrections
    """
    Separated Cpmin_0 model is a very rough fit to Figure 3 of:
    Shademan & Naghib-Lahouti, "Effects of aspect ratio and inclination angle on aerodynamic loads of a flat 
    plate", Advances in Aerodynamics. 
    https://www.researchgate.net/publication/342316140_Effects_of_aspect_ratio_and_inclination_angle_on_aerodynamic_loads_of_a_flat_plate
    Below is a function that computes the critical Mach number from the incompressible Cp_min.
    It's based on a Laitone-rule compressibility correction (similar to Prandtl-Glauert or Karman-Tsien, 
    but higher order), together with the Cp_sonic relation. When the Laitone-rule Cp equals Cp_sonic, we have reached
    the critical Mach number.
    This approach does not admit explicit solution for the Cp0 -> M_crit relation, so we instead regress a 
    relationship out using symbolic regression. In effect, this is a curve fit to synthetic data.
    See fits at: /AeroSandbox/studies/MachFitting/CriticalMach/
    """
    Cpmin_0 = _blend(is_separated, -1 - 0.5 * np.sin(np.radians(alpha)) ** 2, Cpmin_0)
    Cpmin_0 = min_fn(Cpmin_0, np.zeros_like(Cpmin_0), softness=0.001)
    mach_crit = (1.0115710267016 - Cpmin_0 + 0.65824313510071 * (-Cpmin_0) ** 0.6724789439840343) ** -0.5504677038358711
    mach_dd = mach_crit + (0.1 / 320) ** (1 / 3)  # drag divergence taken from W.H. Mason's Korn Equation
    beta_squared_ideal = 1 - mach**2
    beta = max_fn(beta_squared_ideal, -beta_squared_ideal, softness=0.5) ** 0.5
    CL = CL / beta
    CM = CM / beta
    Cpmin = Cpmin_0 / beta  # Prandtl-Glauert

    # Compressibility buffeting and supersonic corrections
    buffet_factor = _blend(
        50 * (mach - (mach_dd + 0.04)),
        _blend((mach - 1) / 0.1, 1, 0.5),
        1
    )
    cla_supersonic_ratio_factor = _blend(
        (mach - 1) / 0.1,
        4 / (2 * np.pi),
        1,
    )
    CL = CL * buffet_factor * cla_supersonic_ratio_factor
    CD_wave = np.where(
        mach < mach_crit,
        0,
        np.where(
            mach < mach_dd,
            80 * (mach - mach_crit) ** 4,
            np.where(
                mach < 1.1,
                _cosine_hermite_patch(
                    mach,
                    x_a=mach_dd,
                    x_b=1.1,
                    f_a=80 * (0.1 / 320) ** (4 / 3),
                    f_b=0.8 * wave_drag_foil_thickness,
                    dfdx_a=0.1,
                    dfdx_b=-0.8 * wave_drag_foil_thickness * 8,
                ),
                _blend(
                    8 * 2 * (mach - 1.1) / (1.2 - 0.8),
                    0.8 * 0.8 * wave_drag_foil_thickness,
                    1.2 * 0.8 * wave_drag_foil_thickness,
                )
            )
        )
    )
    CD = CD + CD_wave

    # Shift aerodynamic center
    has_aerodynamic_center_shift = (mach - (mach_dd + 0.06)) / 0.06
    has_aerodynamic_center_shift = max_fn(
        is_separated,
        has_aerodynamic_center_shift,
        softness=0.1
    )

    CM = CM + _blend(
        has_aerodynamic_center_shift,
        -0.25 * np.cos(np.radians(alpha)) * CL - 0.25 * np.sin(np.radians(alpha)) * CD,
        0,
    )

    return {
        "analysis_confidence": nf_aero["analysis_confidence"],
        "CL": CL,
        "CD": CD,
        "CM": CM,
        "Cpmin": Cpmin,
        "Top_Xtr": Top_Xtr,
        "Bot_Xtr": Bot_Xtr,
        "mach_crit": mach_crit,
        "mach_dd": mach_dd,
        "Cpmin_0": Cpmin_0,
        **{f"upper_bl_theta_{i}": nf_aero[f"upper_bl_theta_{i}"] for i in range(NUM_BL_POINTS)},
        **{f"upper_bl_H_{i}": nf_aero[f"upper_bl_H_{i}"] for i in range(NUM_BL_POINTS)},
        **{f"upper_bl_ue/vinf_{i}": nf_aero[f"upper_bl_ue/vinf_{i}"] for i in range(NUM_BL_POINTS)},
        **{f"lower_bl_theta_{i}": nf_aero[f"lower_bl_theta_{i}"] for i in range(NUM_BL_POINTS)},
        **{f"lower_bl_H_{i}": nf_aero[f"lower_bl_H_{i}"] for i in range(NUM_BL_POINTS)},
        **{f"lower_bl_ue/vinf_{i}": nf_aero[f"lower_bl_ue/vinf_{i}"] for i in range(NUM_BL_POINTS)},
    }
