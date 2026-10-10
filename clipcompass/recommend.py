import torch

from .model import Recommender


def recommend(checkpoint, user_id=0, k=5):
    if k < 1:
        raise ValueError("k must be positive")
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not 0 <= user_id < bundle["config"]["users"]:
        raise ValueError("unknown user ID")
    model = Recommender(**bundle["config"])
    model.load_state_dict(bundle["state"])
    model.eval()
    with torch.no_grad():
        items = model.item_vectors(bundle["text"], bundle["visual"])
        user = model.user_vectors(torch.tensor([user_id]), bundle["history"][user_id:user_id+1])
        retrieval = (user @ items.T)[0]
        retrieval[bundle["seen"][user_id]] = -torch.inf
        count = min(max(k, 50), int(torch.isfinite(retrieval).sum()))
        candidates = torch.topk(retrieval, count).indices
        logits = model.logits(user.expand(count, -1), items[candidates])
        utility = .65*logits[:, 0].sigmoid() + .35*logits[:, 1].sigmoid()
        picked = []
        for _ in range(min(k, count)):
            mmr = utility.clone()
            if picked:
                similarities = items[candidates] @ items[candidates[picked]].T
                mmr -= .15*similarities.max(dim=1).values
                mmr[picked] = -torch.inf
            picked.append(int(mmr.argmax()))
    cards = []
    for pos in picked:
        i = int(candidates[pos])
        c = bundle["catalog"][i]
        cards.append({"item_id": i, "caption": c["caption"], "language": c["language"],
            "engagement_probability": round(float(logits[pos, 0].sigmoid()), 4),
            "completion_probability": round(float(logits[pos, 1].sigmoid()), 4),
            "evidence": {"source_item_id": i, "source_caption": c["caption"]},
            "creative_brief": f"围绕「{c['caption']}」制作自己的三步教程：开场展示目标、演示一个关键步骤、总结适用场景。",
            "generation": "evidence-grounded fixed template; NOT LLM output"})
    return {"user_id": user_id, "source": bundle["provenance"]["source"], "pipeline": "two tower top50 -> multitask ranker -> MMR", "cards": cards}
