"""CPU checks: SE(3), learned ego head, checkpoint compatibility and no GT fallback."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from brain import Brain, EgoHead, ego_objective


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    b = 2
    head = EgoHead(128)
    memory = torch.randn(b, 25, 128)
    times = torch.tensor([[-2., -1.5, -1., -.5, 0.]]).expand(b, -1)
    history = torch.eye(4).expand(b, 5, 4, 4).clone()
    history[:, :, 0, 3] = -times * 2  # 2 m/s forward
    pred = head(memory, history, times)
    assert torch.allclose(pred[:, :, 0, 3], torch.arange(1., 7.).expand(b, -1))
    assert torch.allclose(pred[..., :3, :3], torch.eye(3).expand(b, 6, 3, 3))
    truth = pred.detach().clone()
    truth[:, :, 1, 3] = torch.arange(1., 7.) * .2
    theta = torch.tensor(.1)
    rot = torch.tensor([[theta.cos(), -theta.sin(), 0], [theta.sin(), theta.cos(), 0], [0, 0, 1.]])
    truth[..., :3, :3] = rot
    initial = float(ego_objective(pred, truth)[0].detach())
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    for _ in range(80):
        loss, _ = ego_objective(head(memory, history, times), truth)
        opt.zero_grad(); loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
        opt.step()
    pred = head(memory, history, times)
    final = float(ego_objective(pred, truth)[0].detach())
    assert final < initial * .2, (initial, final)
    r = pred[..., :3, :3]
    assert torch.allclose(r.transpose(-1, -2) @ r, torch.eye(3).expand_as(r), atol=1e-5)
    assert torch.allclose(torch.linalg.det(r), torch.ones(b, 6), atol=1e-5)
    assert torch.allclose(pred[..., 3, :], torch.tensor([0., 0., 0., 1.]).expand(b, 6, 4))

    # Existing actor checkpoint remains loadable and actor inference is unchanged.
    old = Brain().eval()
    new = Brain(predict_ego=True).eval()
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert missing and all(k.startswith('ego_head.') for k in missing) and not unexpected
    x = dict(occ=torch.full((1, 5, 200, 200, 16), 17, dtype=torch.uint8),
             current_to_history=history[:1], history_times=times[:1])
    with torch.no_grad():
        before, after = old(**x), new(**x)
    for k in before:
        assert torch.equal(before[k], after[k]), k
    assert after['ego_future_to_current'].shape == (1, 6, 4, 4)
    restored = Brain(predict_ego=True).eval(); restored.load_state_dict(new.state_dict())
    with torch.no_grad():
        assert torch.equal(restored(**x)['ego_future_to_current'], after['ego_future_to_current'])

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'occ_cerebellum_20261028'))
    from model import ForecastData, Cerebellum
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)
        np.savez(path/'occ.npz', semantics=np.full((200, 200, 16), 17, dtype=np.uint8))
        data = dict(tokens=['test'], current_paths=[str(path/'occ.npz')],
                    future_paths=[[str(path/'occ.npz')]*6],
                    future_to_current=np.full((1, 6, 4, 4), np.nan, dtype=np.float32),
                    predicted_future_to_current=after['ego_future_to_current'].numpy(),
                    centers=np.zeros((1, 256, 2), np.float32), future=np.zeros((1, 256, 6, 2), np.float32),
                    confidence=np.zeros((1, 256), np.float32))
        torch.save(data, path/'pred.pt')
        inputs, _ = ForecastData(path/'pred.pt', ego_source='predicted')[0]
        assert torch.isfinite(inputs['future_to_current']).all(), 'GT pose leaked into inputs'
        net = Cerebellum().eval()
        with torch.no_grad():
            logits = net(**{k:v[None] for k,v in inputs.items()})
        assert logits.shape == (1, 6, 18, 200, 200, 16) and torch.isfinite(logits).all()
        del data['predicted_future_to_current']; torch.save(data, path/'missing.pt')
        try:
            ForecastData(path/'missing.pt', ego_source='predicted')
        except ValueError:
            pass
        else:
            raise AssertionError('Missing predicted poses silently fell back to GT')
    print(dict(checks='passed', initial_ego_loss=initial, final_ego_loss=final,
               actor_outputs_unchanged=True, predicted_pose_decoder=True, no_GT_fallback=True))


if __name__ == '__main__':
    main()
