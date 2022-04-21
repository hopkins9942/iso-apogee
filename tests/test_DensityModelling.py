import pytest
import torch

from DensityModelling import FeHBinnedDoubleExpPPP as PPP

#TODO: add in tests for data files - checking length of file should be enough

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

oneBin_batched = (
    torch.tensor([-1.0,-0.5]),
    1.5,
    torch.tensor([0.1,0.2,0.3]),
    torch.tensor([[0.4],[0.5],[0.6],[0.7]]),
    )

threeBin_batchedParam = (
    torch.tensor([[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]]),
    1.5,
    0.1,
    torch.tensor([[0.4],[0.5],[0.6],[0.7]]),
    )

threeBin_batchedBin = (
    torch.tensor([[[-1.0 ,-0.5 ],[-0.5 ,0.0 ],[0.0 ,0.5 ]],
                  [[-0.75,-0.25],[-0.25,0.25],[0.25,0.75]]]),
    1.5,
    0.1,
    0.4,
    )

oneBin_broadcastFail = (
    torch.tensor([-0.5,0.0]),
    1.0,
    torch.tensor([0.1,0.2]),
    torch.tensor([0.5,0.6,0.7]),
    )

threeBin_broadcastFail = (
    torch.tensor([[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]]),
    1.5,
    0.1,
    torch.tensor([0.4,0.5]),
    )

threeBin_typeFail = (
    torch.tensor([[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]]),
    1.5,
    0.1,
    [0.4,0.5],
    )


@pytest.mark.parametrize("inputs,expected_attributes",
    [
    (oneBin, (torch.tensor([0.0,0.5]), torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0))),
    (threeBin_sameParam, (torch.tensor([[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]]), torch.tensor([0.0,0.0,0.0]), torch.tensor([1.0,1.0,1.0]), torch.tensor([1.0,1.0,1.0]))),
    (threeBin_oneDiffParam, (torch.tensor([[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]]), torch.tensor([-0.5,0.5,0.0]), torch.tensor([2.0,2.0,2.0]), torch.tensor([0.2,0.2,0.2]))),
    (threeBin_threeDiffParam, (torch.tensor([[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]]), torch.tensor([-0.5,0.5,0]), torch.tensor([0.8,1.0,1.2]), torch.tensor([0.1,0.2,0.3]))),
    (oneBin_threeDiffParam, (torch.tensor([[0.0,0.5],[0.0,0.5],[0.0,0.5]]), torch.tensor([-0.5,0.5,0]), torch.tensor([0.8,1.0,1.2]), torch.tensor([0.1,0.2,0.3]))),
    (oneBin_batched, (torch.tensor([-1.0,-0.5]).expand(4,3,2), 1.5*torch.ones([4,3]), torch.tensor([0.1,0.2,0.3]).expand(4,3), torch.tensor([[0.4],[0.5],[0.6],[0.7]]).expand(4,3))),
    (threeBin_batchedParam, (torch.tensor([[[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]],
                                           [[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]],
                                           [[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]],
                                           [[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]]]),
                            1.5*torch.ones([4,3]), 0.1*torch.ones([4,3]), torch.tensor([[0.4],[0.5],[0.6],[0.7]]).expand(4,3))),
    (threeBin_batchedBin,  (torch.tensor([[[-1.0,-0.5],[-0.5,0.0],[0.0,0.5]],
                                          [[-0.75,-0.25],[-0.25,0.25],[0.25,0.75]]]),
                            1.5*torch.ones([2,3]), 0.1*torch.ones([2,3]), 0.4*torch.ones([2,3]))),
    pytest.param(oneBin_broadcastFail, (), marks=pytest.mark.xfail),
    pytest.param(threeBin_broadcastFail, (), marks=pytest.mark.xfail),
    pytest.param(threeBin_typeFail, (), marks=pytest.mark.xfail),
    ],
    ids=["oneBin","threeBin_sameParam","threeBin_oneDiffParam","threeBin_threeDiffParam","oneBin_threeDiffParam","oneBin_batched","threeBin_batchedParam","threeBin_batchedBin","oneBin_broadcastFail","threeBin_broadcastFail","threeBin_typeFail"
])
def test_distro_broadcasting(inputs, expected_attributes):
    distro = PPP(*inputs)
    attributes = (distro.FeHBinEdges, distro.logA, distro.a_R, distro.a_z)
    assert all(torch.equal(attributes[i], expected_attributes[i]) for i in range(4))

@pytest.mark.parametrize("inputs,expected_shape",
    [
    (oneBin, torch.Size()),
    (threeBin_sameParam, torch.Size([3])),
    (threeBin_oneDiffParam, torch.Size([3])),
    (threeBin_threeDiffParam, torch.Size([3])),
    (oneBin_threeDiffParam, torch.Size([3])),
    (oneBin_batched, torch.Size([4,3])),
    (threeBin_batchedParam, torch.Size([4,3])),
    (threeBin_batchedBin, torch.Size([2,3]))
    ],
    ids=["oneBin","threeBin_sameParam","threeBin_oneDiffParam","threeBin_threeDiffParam","oneBin_threeDiffParam","oneBin_batched","threeBin_batchedParam","threeBin_batchedBin"]
)
def test_distro_batch_shapes(inputs, expected_shape):
    distro = PPP(*inputs)
    assert (distro.batch_shape == expected_shape) and (distro.event_shape == torch.Size([3]))

def test_distro_EffectiveVolume():
    pass
