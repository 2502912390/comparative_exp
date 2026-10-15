"""Author Algorithm 1 / eval_for_multi with a separate replayable CPU RNG."""
import numpy as np
import torch


def initial_rng(seed):
    return torch.Generator(device="cpu").manual_seed(seed).get_state()


@torch.no_grad()
def monte_carlo(model, data, node_ids, rng_state, *, keep_predictions=False):
    weak, strong, confidence = model.inference_parts(data)
    if not all(torch.isfinite(value).all() for value in (weak, strong, confidence)):
        raise FloatingPointError("Nonfinite Mowst expert probabilities or gate")
    ids = node_ids.to(data.x.device)
    first = weak.argmax(-1)
    second = strong.argmax(-1)
    predictions, routes, counts = [], [], []
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(rng_state.cpu())
        for _ in range(model.args.m_times):
            # Preserve author CPU torch.rand(...).to(device) shape, '<' rule,
            # and hard expert selection. No probability average/majority vote.
            route = torch.rand(confidence.shape).to(data.x.device) < confidence
            predicted = torch.where(route.view(-1), first, second)
            chosen = predicted[ids]
            counts.append(int((chosen == data.y.view(-1)[ids]).sum().item()))
            if keep_predictions:
                predictions.append(chosen.cpu().numpy())
                routes.append(route.view(-1)[ids].cpu().numpy())
        after = torch.get_rng_state().clone()
    observed = len(node_ids) * model.args.m_times
    result = {"acc": sum(counts) / observed, "correct": sum(counts), "total": observed,
              "unique_test_nodes": len(node_ids), "mc_samples": model.args.m_times,
              "per_draw_correct": counts, "rng_after": after}
    if keep_predictions:
        result.update(predictions=np.stack(predictions), routes=np.stack(routes),
            expert_probabilities=torch.stack((weak[ids], strong[ids])).cpu().numpy(),
            gate_probabilities=confidence.view(-1)[ids].cpu().numpy())
    return result
