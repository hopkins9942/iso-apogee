import pytest
from DensityModelling import FeHBinnedDoubleExpPPP as distr


class TestTests:
    def test_pass(self):
        assert 1==1

    @pytest.mark.xfail
    def test_fail(self):
        assert 1==2

class TestDistribution:
    def test_
