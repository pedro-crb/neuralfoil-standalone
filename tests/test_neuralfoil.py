import pytest
import neuralfoil as nf


def test_basic_functionality():
    aero = nf.get_aero_from_kulfan_parameters(
        alpha=5,
        Re=1e6
    )


if __name__ == '__main__':
    pytest.main()
