import tempfile
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
from dataset import RegistrationDataset
from model import Voxelmorph
from spatial_transformer import SpatialTransformer
from integration import IntegrateVelocityField
from losses import NCCLoss
from metrics import mean_dice, fold_fraction
from checkpoint import load_model
from train import epoch_loss


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_identity_and_translation(self):
        image = torch.zeros(1,1,32,32)
        image[:,:,10:20,10:20] = 1
        for align in [True, False]:
            transformer = SpatialTransformer(align_corners=align)
            torch.testing.assert_close(transformer(image, torch.zeros(1,2,32,32)), image)
            flow = torch.zeros(1,2,32,32)
            flow[:,0] = 3
            expected = torch.zeros_like(image)
            expected[:,:,10:20,7:17] = 1
            torch.testing.assert_close(transformer(image,flow), expected, atol=1e-5, rtol=1e-5)

    def test_nearest_labels(self):
        image = torch.randint(0,5,(1,1,32,32)).float()
        warped = SpatialTransformer("nearest")(image,torch.rand(1,2,32,32))
        self.assertTrue(set(warped.unique().tolist()) <= set(image.unique().tolist()))

    def test_integration_and_gradients(self):
        for steps in [0, 5]:
            velocity = torch.ones(1,2,32,32,requires_grad=True)
            field = IntegrateVelocityField(steps)(velocity)
            torch.testing.assert_close(field, velocity)
            field.square().mean().backward()
            self.assertTrue(torch.isfinite(velocity.grad).all())
            torch.testing.assert_close(IntegrateVelocityField(steps)(torch.zeros_like(velocity)),torch.zeros_like(velocity))

    def test_dataset_distinct_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            moving, fixed = Path(tmp)/"moving", Path(tmp)/"fixed"
            moving.mkdir(); fixed.mkdir()
            Image.fromarray(np.zeros((32,32),dtype=np.uint8)).save(moving/"a.png")
            Image.fromarray(np.full((32,32),255,dtype=np.uint8)).save(fixed/"a.png")
            a,b = RegistrationDataset(moving,fixed)[0]
            self.assertEqual(float(a.max()),0)
            self.assertEqual(float(b.min()),1)

    def test_metrics(self):
        labels = np.array([[0,1],[0,1]])
        self.assertEqual(mean_dice(labels,labels),1)
        self.assertEqual(fold_fraction(np.zeros((2,32,32))),0)
        yy,xx = np.indices((32,32))
        flow = np.stack((-2*xx,np.zeros_like(yy)))
        self.assertEqual(fold_fraction(flow),1)

    def test_train_updates_validation_does_not(self):
        model = Voxelmorph(3)
        transformer = SpatialTransformer()
        loader = DataLoader(TensorDataset(torch.rand(2,1,32,32),torch.rand(2,1,32,32)),batch_size=1)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
        before = model.flow_head.flow.weight.detach().clone()
        epoch_loss(model,transformer,loader,torch.device("cpu"),0.005,optimizer)
        self.assertFalse(torch.equal(before,model.flow_head.flow.weight))
        before = model.flow_head.flow.weight.detach().clone()
        epoch_loss(model,transformer,loader,torch.device("cpu"),0.005)
        self.assertTrue(torch.equal(before,model.flow_head.flow.weight))

    def test_odd_shape_and_checkpoint(self):
        a,b = torch.rand(1,1,33,39),torch.rand(1,1,33,39)
        model = Voxelmorph(5).eval()
        with torch.no_grad(): expected = model(a,b)
        self.assertEqual(tuple(expected.shape),(1,2,33,39))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.pt"
            torch.save(dict(model_state=model.state_dict(),settings={"integration_steps":5}),path)
            restored,_ = load_model(path,torch.device("cpu"))
            with torch.no_grad(): actual = restored(a,b)
            torch.testing.assert_close(expected,actual)

    def test_ncc_backward(self):
        a = torch.rand(1,1,32,32,requires_grad=True)
        loss = NCCLoss()(a,torch.rand_like(a))
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())


if __name__ == "__main__":
    unittest.main()
