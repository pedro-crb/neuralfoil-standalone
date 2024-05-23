import pytest
import numpy as np
import neuralfoil_standalone as nf


foil_kulfan_parameters = {
    'lower_weights': np.array([-0.22714532, -0.15717196,  0.07265458,  0.27418843,
                               -0.10665784, 0.53098821,  0.04162837, 0.6919114]),
    'upper_weights': np.array([0.22291517, 0.06650672, 0.6873575, 0.11412786,
                               0.29926671, 0.60351015, 0.05685255, 0.88945245]),
    'TE_thickness': 0.000504877343148637,
    'leading_edge_weight': 0.8042953132553561
}


def test_basic_functionality():
    aero = nf.get_aero_from_kulfan_parameters(
        kulfan_parameters=foil_kulfan_parameters,
        alpha=5,
        Re=1e6,
    )
    print('\n')
    print(aero)


if __name__ == '__main__':
    pytest.main()
