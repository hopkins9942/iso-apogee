import pytest
import torch

from DensityModelling import FeHBinnedDoubleExpPPP as PPP


class TestTests:
    def test_pass(self):
        assert 1==1

    @pytest.mark.xfail
    def test_fail(self):
        assert 1==2


# distro instance parameters for use in tests
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

#TODO add example with more than one batch dimention

#@pytest.mark.parametrize("inputs", [
#    oneBin,
#    threeBin_sameParam,
#    threeBin_oneDiffParam,
#    threeBin_threeDiffParam,
#    oneBin_threeDiffParam,
#    ],
#    ids=["oneBin","threeBin_sameParam","threeBin_oneDiffParam","threeBin_threeDiffParam","oneBin_threeDiffParam"
#])
#def test_distro_binEdges(inputs):
#    assert torch.equal(inputs[0], PPP(*inputs).FeHBinEdges)


@pytest.mark.parametrize("inputs,expected_attributes",
    [
    (oneBin, (torch.tensor([0.0,0.5]), torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0))),
    (threeBin_sameParam, (torch.tensor([[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]]), torch.tensor([0.0,0.0,0.0]), torch.tensor([1.0,1.0,1.0]), torch.tensor([1.0,1.0,1.0]))),
    (threeBin_oneDiffParam, (torch.tensor([[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]]), torch.tensor([-0.5,0.5,0.0]), torch.tensor([2.0,2.0,2.0]), torch.tensor([0.2,0.2,0.2]))),
    (threeBin_threeDiffParam, (torch.tensor([[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]]), torch.tensor([-0.5,0.5,0]), torch.tensor([0.8,1.0,1.2]), torch.tensor([0.1,0.2,0.3]))),
    (oneBin_threeDiffParam, (torch.tensor([[0.0,0.5],[0.0,0.5],[0.0,0.5]]), torch.tensor([-0.5,0.5,0]), torch.tensor([0.8,1.0,1.2]), torch.tensor([0.1,0.2,0.3]))),
    ],
    ids=["oneBin","threeBin_sameParam","threeBin_oneDiffParam","threeBin_threeDiffParam","oneBin_threeDiffParam"
])
def test_distro_broadcasting(inputs, expected_attributes):
    distro = PPP(*inputs)
    attributes = (distro.FeHBinEdges, distro.logA, distro.a_R, distro.a_z)
    print(attributes)
    assert all(torch.equal(attributes[i], expected_attributes[i]) for i in range(4))

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

