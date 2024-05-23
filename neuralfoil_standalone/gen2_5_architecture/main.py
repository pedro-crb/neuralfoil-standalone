try:
    import optisandbox.numpy as np
    from optisandbox.numpy import length
except ImportError:
    import numpy as np
    length = len

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


def get_aero_from_kulfan_parameters(
        kulfan_parameters: Dict[str, Union[float, np.ndarray]],
        alpha: Union[float, np.ndarray],
        Re: Union[float, np.ndarray],
        n_crit: Union[float, np.ndarray] = 9.0,
        xtr_upper: Union[float, np.ndarray] = 1.0,
        xtr_lower: Union[float, np.ndarray] = 1.0,
        model_size="large"
) -> Dict[str, Union[float, np.ndarray]]:

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
