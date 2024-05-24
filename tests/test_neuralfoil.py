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


def test_corrections():
    params = np.array([
        [10, 100000, 0.0, 3.0, 0.7, 0.1, 0.1, 7],
        [11, 200000, 0.0, 2.0, 0.7, 0.2, 0.2, 9],
        [12, 300000, 0.0, 1.0, 0.7, 0.3, 0.3, 11]
    ])

    aero = nf.get_aero_with_corrections(
        kulfan_parameters=foil_kulfan_parameters,
        alpha=params[:, 0],
        Re=params[:, 1],
        mach=params[:, 2],
        n_crit=params[:, 7],
        xtr_upper=params[:, 5],
        xtr_lower=params[:, 6],
        model_size='large',
        control_surface_deflection=params[:, 3],
        control_surface_hinge_point=1-params[:, 4],
        wave_drag_foil_thickness=0.12,
    )
    print('\n')
    print(aero)


if __name__ == '__main__':
    pytest.main()
