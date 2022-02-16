import pytest
import torch

from DensityModelling import FeHBinnedDoubleExpPPP as PPP


class TestTests:
    def test_pass(self):
        assert 1==1

    @pytest.mark.xfail
    def test_fail(self):
        assert 1==2


# distro instances for use in tests
oneBin = (torch.tensor([0.0,0.5]), 0, 1, 1)

threeBin_sameParam = (
    torch.tensor([[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]]), 0, 1, 1
    )

threeBin_oneDiffParam = (
    torch.tensor([[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]]),
    torch.tensor([-0.5,0.5,0]),
    2,
    0.2
    )

threeBin_threeDiffParam = (
    torch.tensor([[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]]),
    torch.tensor([-0.5,0.5,0]),
    torch.tensor([0.8,1,1.2]),
    torch.tensor([0.1,0.2,0.3])
    )

oneBin_threeDiffParam = (
    torch.tensor([0.0,0.5]),
    torch.tensor([-0.5,0.5,0]),
    torch.tensor([0.8,1,1.2]),
    torch.tensor([0.1,0.2,0.3])
    )

@pytest.mark.parametrize("inputs", [
    oneBin,
    threeBin_sameParam,
    threeBin_oneDiffParam,
    threeBin_threeDiffParam,
    oneBin_threeDiffParam,
    ],
    ids=["oneBin","threeBin_sameParam","threeBin_oneDiffParam","threeBin_threeDiffParam","oneBin_threeDiffParam"
])
def test_distro_binEdges(inputs):
    assert torch.equal(inputs[0], PPP(*inputs).FeHBinEdges)


@pytest.mark.parametrize("inputs,expected_param_shape", [
    (oneBin, torch.Size([1])),
    (threeBin_sameParam, torch.Size([3])),
    (threeBin_oneDiffParam, torch.Size([3])),
    (threeBin_threeDiffParam, torch.Size([3])),
    (oneBin_threeDiffParam, torch.Size([3])),
    ],
    ids=["oneBin","threeBin_sameParam","threeBin_oneDiffParam","threeBin_threeDiffParam","oneBin_threeDiffParam"
])
def test_distro_shapes(inputs, expected_param_shape):
    distro = PPP(*inputs)
    shapes = (distro.logA.shape, distro.a_R.shape, distro.a_z.shape)
    assert all(shapes[i]==expected_param_shape)

#@pytest.mark.parametrize("bins,logA,a_R,a_z", [
#    (
#    torch.tensor([0.0,0.5]), 0, 1, 1
#    ), # one bin
#    (
#    torch.tensor([[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]]), 0, 1, 1
#    ), # three bins, same parameters
#    (
#    torch.tensor([[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]]),
#    torch.tensor([-0.5,0.5,0]),
#    2,
#    0.2
#    ), # three bins, one different parameter
#    (
#    torch.tensor([[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]]),
#    torch.tensor([-0.5,0.5,0]),
#    torch.tensor([0.8,1,1.2]),
#    torch.tensor([0.1,0.2,0.3])
#    ), # three bins, all different parameters
#]) # possible additions - one bin, multiple parameters?
#class TestDistribution:
#    #distro = PPP(bins,logA,a_R,a_z) # should/can this be a fixture?
#
#    def test_bins(self, bins, logA, a_R, a_z):
#        assert torch.equal(bins, PPP(bins, logA, a_R, a_z).FeHBinEdges)
#
#
#    @pytest.fixtures
#    def 1bin(self):
#        return PPP(torch.tensor([0.0,0.5]), 0, 1, 1)
#
#    @pytest.fixtures
#    def 3bin_alldiffparams(self):
#        return PPP(torch.tensor([[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]]),
#                   torch.tensor([-0.5,0.5,0]),
#                   torch.tensor([0.8,1,1.2]),
#                   torch.tensor([0.1,0.2,0.3]))
#
#    @pytest.fixtures
#    def 3bin_onediffparams(self):
#        return PPP(torch.tensor([[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]]),
#                   torch.tensor([-0.5,0.5,0]),
#                   2,
#                   0.2)
#
#    @pytest.fixtures
#    def 3bin_sameparams(self):
#        return PPP(torch.tensor([[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]]),
#                   0,
#                   1,
#                   1)

