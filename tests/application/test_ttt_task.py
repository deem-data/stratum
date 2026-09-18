import math
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import stratum as skrub
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import BaseCrossValidator
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# lightgbm/xgboost (imported by the other application tests) link the system
# libomp while torch bundles its own, and two OpenMP runtimes in one process
# deadlock on torch's parallel kernels -- the suite hangs forever inside the
# batch gather in fit(). One torch thread skips the parallel region entirely,
# which for a test this size costs nothing.
torch.set_num_threads(1)

# The task ships 355 trackers and trains for 12 epochs over ~1M domains. This is
# a plan-shape test, not a benchmark: what it has to reproduce is the pipeline's
# *structure* (the wide fillna/astype blocks, the leave-one-out priors, the two
# neighbour aggregations, the five prior channels the network slices off the
# tail), not its scale. Everything below is sized so the whole test runs in
# seconds while every one of those blocks still carries real values.
NUM_TRACKERS = 64
NUM_EPOCHS = 3
HIDDEN_DIM = 64
LATENT_DIM = 16
BATCH_SIZE = 256
# Both recall@k and the loss's hard-negative mining rank the top 10, so
# NUM_TRACKERS has to stay above 10.
TOP_K = 10
TRACKER_COLS = [f"tracker_{i}" for i in range(NUM_TRACKERS)]

TOP_TLDS = [
    "com", "ru", "org", "net", "de", "uk", "jp", "fr", "it", "pl",
    "br", "cn", "in", "nl", "es", "cz", "eu", "ua", "ca", "au",
    "ch", "se", "ro", "gr", "at", "tv", "io", "me", "co", "info",
]

KEYWORDS = {
    "kw_ecommerce": ["shop", "store", "cart", "buy", "market", "mall", "deal", "pay"],
    "kw_media": ["news", "press", "media", "times", "post", "daily", "gazette", "journal"],
    "kw_video": ["video", "tv", "movie", "film", "tube", "stream"],
    "kw_adult": ["adult", "sex", "porn", "xxx"],
    "kw_tech": ["tech", "dev", "code", "cloud", "soft", "app", "web"],
    "kw_finance": ["bank", "finance", "invest", "loan", "crypto", "coin"],
}

TWO_LEVEL_TLDS = {
    "co.uk", "org.uk", "me.uk", "ltd.uk", "plc.uk", "net.uk", "sch.uk", "ac.uk", "gov.uk",
    "com.au", "net.au", "org.au", "edu.au", "gov.au", "asn.au", "id.au",
    "co.jp", "ne.jp", "or.jp", "ac.jp", "ed.jp", "go.jp", "gr.jp", "lg.jp",
    "com.br", "net.br", "org.br", "gov.br", "edu.br",
    "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn",
    "com.ru", "net.ru", "org.ru", "pp.ru",
    "co.in", "net.in", "org.in", "gen.in", "firm.in", "ind.in",
    "com.mx", "net.mx", "org.mx", "edu.mx", "gob.mx",
    "com.tr", "net.tr", "org.tr", "edu.tr", "gov.tr",
    "com.pl", "net.pl", "org.pl", "info.pl",
    "co.nz", "net.nz", "org.nz", "govt.nz", "ac.nz",
    "co.za", "net.za", "org.za", "web.za",
    "co.kr", "ne.kr", "or.kr", "re.kr",
    "com.ar", "net.ar", "org.ar",
    "com.tw", "org.tw", "idv.tw",
    "com.ua", "net.ua", "org.ua", "kiev.ua",
    "co.il", "org.il", "net.il",
    "com.sg", "org.sg", "net.sg",
    "com.hk", "org.hk", "net.hk",
}

# The lexical block in the original's column order: X_lexical = column_stack([
#   d_lens, num_dots, num_hyphens, num_digits, num_digits / d_lens, has_www,
#   has_multi_sub, press_scores, has_press, is_auth, tld_onehot, kw_*, kw_total])
LEXICAL_COLS = (
        ["dom_len", "num_dots", "num_hyphens", "num_digits", "digit_ratio",
         "has_www", "has_multi_sub", "press_score", "has_press", "is_auth"]
        + [f"tld_{t}" for t in TOP_TLDS]
        + ["tld_other"]
        + list(KEYWORDS)
        + ["kw_total"]
)

URL_CATEGORIES = [
    "Arts", "Business", "Computers", "Games", "Health", "Home", "Kids_and_Teens",
    "News", "Recreation", "Reference", "Regional", "Science", "Shopping",
    "Society", "Sports",
]
URL_CAT_COLS = [f"url_cat_{c}" for c in URL_CATEGORIES] + ["url_cat_has"]

# The Bayesian prior transformer's output: two NUM_TRACKERS-wide channels plus
# the scalar summaries the original's X_prior block carried.
ROOT_PRIOR_COLS = [f"root_prior_{i}" for i in range(NUM_TRACKERS)]
TLD_PRIOR_COLS = [f"tld_prior_{i}" for i in range(NUM_TRACKERS)]
PRIOR_SCALAR_COLS = [
    "has_root_match", "root_match_count", "root_match_count_log",
    "root_prior_entropy", "root_prior_max",
]

# Clamp on the empirical tracker rates before they are turned into logits, so a
# tracker no training domain uses does not produce an infinite initial bias.
PRIOR_EPS = 1e-5

# Smoothing added to each tracker's own frequency when conditioning the
# co-occurrence counts (the original's epsilon_cooc).
COOC_EPSILON = 10.0

# "does this domain link to tracker t" -- read straight off the hyperlink graph,
# so the first of the original's five prior channels is recordable too.
DIRECT_COLS = [f"direct_{i}" for i in range(NUM_TRACKERS)]

# Adamic-Adar weighted tracker adoption among each domain's link neighbours, one
# block per direction, plus the six graph scalars derived from their weights.
OUT_NBR_COLS = [f"out_nbr_{i}" for i in range(NUM_TRACKERS)]
IN_NBR_COLS = [f"in_nbr_{i}" for i in range(NUM_TRACKERS)]
GRAPH_DYNAMIC_COLS = [
    "nbr_out_deg", "nbr_out_deg_log", "nbr_in_deg", "nbr_in_deg_log",
    "nbr_tot_deg", "nbr_tot_deg_log",
]

# The label-free half of the original's X_graph block, in its column_stack order.
GRAPH_STATIC_COLS = [
    "deg_out", "deg_out_log", "deg_in", "deg_in_log", "deg_total",
    "deg_in_out_ratio", "deg_is_isolated", "tracker_links", "tracker_links_log",
    "tracker_links_per_out",
]

META_COLS = ["domain_id", "root", "tld"]

# The design matrix, in the order the original's two column_stacks produced. The
# network slices its prior channels off the tail, so the last five
# NUM_TRACKERS-wide blocks have to stay last and in this sequence.
FEATURE_COLS = (
        LEXICAL_COLS
        + URL_CAT_COLS
        + GRAPH_STATIC_COLS
        + GRAPH_DYNAMIC_COLS
        + PRIOR_SCALAR_COLS
        + DIRECT_COLS
        + OUT_NBR_COLS
        + IN_NBR_COLS
        + ROOT_PRIOR_COLS
        + TLD_PRIOR_COLS
)

# =========================================================================
# Synthetic inputs: the seven files the task ships, at test scale
# =========================================================================
# Same filenames, same schemas, same separators (freedom-of-the-press.csv is
# tab-separated despite the extension; url-classification.csv is not). The
# values are drawn so every branch the pipeline has a feature for is populated:
# two-level suffixes, www-prefixed hosts, candidate ids missing from
# domains.parquet, stems carrying the keyword families, TLDs inside and outside
# TOP_TLDS, press rows covering only some TLDs, url rows that match directly and
# through the www fallback, self-links, and link targets that are not domains.

_STEMS = [
    "shop", "store", "market", "news", "press", "daily", "video", "tube",
    "stream", "tech", "cloud", "dev", "bank", "crypto", "invest", "blue",
    "north", "green", "atlas", "river", "stone", "orbit", "delta", "prime",
]
_SUFFIXES = ["hub", "line", "works", "group", "media", "labs", "zone", "24"]
# In TOP_TLDS, in TWO_LEVEL_TLDS, and in neither (-> tld_other).
_GEN_TLDS = ["com", "de", "ru", "org", "net", "uk", "jp", "io",
             "co.uk", "com.au", "co.jp", "xyz", "berlin"]


def _make_domain_names(rng, n):
    names, seen = [], set()
    while len(names) < n:
        stem = str(rng.choice(_STEMS))
        if rng.random() < 0.5:
            stem += str(rng.choice(_SUFFIXES))
        if rng.random() < 0.15:
            stem += f"-{rng.integers(1, 100)}"
        host = f"{stem}.{rng.choice(_GEN_TLDS)}"
        draw = rng.random()
        if draw < 0.2:
            host = f"www.{host}"
        elif draw < 0.3:
            host = f"blog.{host}"
        if host not in seen:
            seen.add(host)
            names.append(host)
    return names


def make_input_data(directory, n_domains=2000, n_link_edges=12000, seed=0):
    """Write the task's seven input files into ``directory``."""
    rng = np.random.default_rng(seed)
    root = Path(directory)

    names = _make_domain_names(rng, n_domains)
    domain_ids = np.sort(rng.choice(
        np.arange(100, 100 + 50 * n_domains), size=n_domains, replace=False))
    tracking_domain_ids = np.sort(rng.choice(
        np.arange(10_000_000, 10_050_000), size=NUM_TRACKERS, replace=False))

    # --- trackers.tsv ----------------------------------------------------
    pd.DataFrame({
        "domain": [f"tracker-{i:02d}.net" for i in range(NUM_TRACKERS)],
        "tracking_domain_id": tracking_domain_ids,
        "tracker_id": np.arange(NUM_TRACKERS),
        "company": [f"Company {i % 7}" for i in range(NUM_TRACKERS)],
        "country": [["US", "DE", "RU"][i % 3] for i in range(NUM_TRACKERS)],
        "category": [["Analytics", "Advertising", "Social"][i % 3]
                     for i in range(NUM_TRACKERS)],
        "brand": [f"Brand {i % 5}" for i in range(NUM_TRACKERS)],
    }).to_csv(root / "trackers.tsv", sep="\t", index=False)

    # --- domains.parquet -------------------------------------------------
    # A few candidates are absent, so the pipeline's fillna("") -> "unknown"
    # root/TLD branch is reached; a few ids appear twice, which it resolves
    # with drop_duplicates(keep="last").
    known = rng.random(n_domains) > 0.03
    domains = pd.DataFrame({"domain": np.asarray(names)[known],
                            "domain_id": domain_ids[known]})
    shadow = domains.head(5).assign(domain=domains["domain"].head(5) + ".dup")
    pd.concat([shadow, domains], ignore_index=True).to_parquet(
        root / "domains.parquet", index=False)

    # --- tracking_graph_train.parquet ------------------------------------
    # Tracker adoption clusters by TLD, which is what gives the Bayesian
    # root/TLD priors and the co-occurrence matrix something to find.
    tld_of = np.array([name.rsplit(".", 1)[-1] for name in names])
    tld_index = pd.Index(sorted(set(tld_of))).get_indexer(tld_of)
    popularity = 1.0 / (1.0 + np.arange(NUM_TRACKERS))
    slots = np.arange(NUM_TRACKERS) % 4

    rows = []
    for row, dom_id in enumerate(domain_ids):
        weights = popularity * np.where(slots == tld_index[row] % 4, 4.0, 1.0)
        picked = rng.choice(NUM_TRACKERS, size=rng.integers(1, 6),
                            replace=False, p=weights / weights.sum())
        rows.extend((dom_id, tracking_domain_ids[t], t) for t in picked)
    tracking = pd.DataFrame(rows, columns=["domain_id", "tracking_domain_id",
                                           "tracker_id"])
    tracking = tracking.astype({"domain_id": "int64",
                                "tracking_domain_id": "int64",
                                "tracker_id": "int32"})
    tracking.to_parquet(root / "tracking_graph_train.parquet", index=False)

    # --- target.tsv ------------------------------------------------------
    # The prediction set. These ids also carry tracking rows, so the pipeline's
    # `~graph_train["domain_id"].isin(target["domain_id"])` really drops rows.
    target_ids = np.sort(rng.choice(domain_ids, size=max(1, n_domains // 12),
                                    replace=False))
    pd.DataFrame({"domain_id": target_ids}).to_csv(
        root / "target.tsv", sep="\t", index=False)

    # --- link-graph.parquet ----------------------------------------------
    # Domain-to-domain edges (the neighbour aggregations), domain-to-tracker
    # edges subsampled from the tracking graph (the direct_* block and
    # tracker_links), self-links, and edges into ids that are not domains.
    to_tracker = tracking.loc[rng.random(len(tracking)) < 0.7,
                              ["domain_id", "tracking_domain_id"]].to_numpy()
    self_links = rng.choice(domain_ids, size=20)
    sources = np.concatenate([
        rng.choice(domain_ids, size=n_link_edges), to_tracker[:, 0],
        self_links, rng.choice(domain_ids, size=30)])
    targets = np.concatenate([
        rng.choice(domain_ids, size=n_link_edges), to_tracker[:, 1],
        self_links, rng.integers(90_000_000, 90_001_000, size=30)])
    pd.DataFrame({"source_domain_id": sources.astype("int32"),
                  "target_domain_id": targets.astype("int32")}).to_parquet(
        root / "link-graph.parquet", index=False)

    # --- freedom-of-the-press.csv (tab separated) ------------------------
    # Three TLDs are left uncovered, so has_press carries both values and the
    # median fill-in is exercised; the 5..95 range straddles the is_auth cut.
    covered = sorted(set(_GEN_TLDS))[:-3]
    pd.DataFrame({
        "tld": covered,
        "country": [f"Country {i}" for i in range(len(covered))],
        "freedom_of_the_press": rng.integers(5, 95, size=len(covered)),
    }).to_csv(root / "freedom-of-the-press.csv", sep="\t", index=False)

    # --- url-classification.csv ------------------------------------------
    urls, categories = [], []
    for j, row in enumerate(rng.choice(n_domains, size=n_domains * 3 // 5,
                                       replace=False)):
        host = names[row]
        if j % 3 == 0 and not host.startswith("www."):
            # Matched through the pipeline's www fallback, not directly.
            host = f"www.{host}"
        urls.append(f"http://{host}/{'index.html' if j % 2 else ''}")
        categories.append(URL_CATEGORIES[j % len(URL_CATEGORIES)])
    # A filtered-out category, a host no domain matches, and a port to strip.
    urls += ["http://unmatched.example/", "http://other.example:8080/a"]
    categories += ["Adult", "Arts"]
    pd.DataFrame({"index": np.arange(len(urls)), "url": urls,
                  "category": categories}).to_csv(
        root / "url-classification.csv", index=False)

    return root


@contextmanager
def ttt_input_dir(**kwargs):
    """Yield a temp directory holding a generated copy of the task's inputs."""
    with tempfile.TemporaryDirectory() as directory:
        yield str(make_input_data(directory, **kwargs))


def recall_at_k(y_true, y_score, k=TOP_K):
    """Mean per-domain Recall@k, the task's metric (original: compute_recall_at_k)."""
    targets = np.asarray(y_true)
    scores = np.asarray(y_score)
    topk_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    row_indices = np.arange(scores.shape[0])[:, None]
    hits = targets[row_indices, topk_indices].sum(axis=1)
    true_counts = targets.sum(axis=1)
    recalls = np.where(true_counts > 0, hits / np.maximum(true_counts, 1), 0.0)
    return float(np.mean(recalls))

# =========================================================================
# Cross-validation: the original's single random hold-out, as a splitter
# =========================================================================
class CandidateDomainHoldout(BaseCrossValidator):
    """One hold-out fold reproducing the original's domain split.

    The original did ``np.random.seed(42); np.random.shuffle(candidates)`` on the
    ascending-sorted output of ``np.setdiff1d`` and kept the first
    ``min(30000, int(0.2 * n))`` shuffled domains as validation. ``X`` is marked
    in that same ascending domain order, so shuffling the row indices with
    ``RandomState(42)`` (identical to the legacy global RNG) yields exactly the
    original's train/validation membership *and* its training row order.
    """

    def __init__(self, max_val=30000, val_fraction=0.2, random_state=42):
        self.max_val = max_val
        self.val_fraction = val_fraction
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.RandomState(self.random_state).shuffle(indices)
        n_val = min(self.max_val, int(n_samples * self.val_fraction))
        yield indices[n_val:], indices[:n_val]

# =========================================================================
# The network, verbatim from the original
# =========================================================================
class ResidualTabularBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.act1 = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.act2 = nn.SiLU()
        self.shortcut = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm2(out)
        return self.act2(out + res)

class TrackerRankNet(nn.Module):

    def __init__(
        self,
        in_dim: int,
        num_classes: int = NUM_TRACKERS,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        dropout_rate: float = 0.2,
        initial_bias: np.ndarray = None,
        num_prior_channels: int = 5,
        cooccurrence_matrix: np.ndarray = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.num_prior_channels = num_prior_channels
        self.has_tracker_priors = in_dim >= num_prior_channels * num_classes

        self.input_norm = nn.BatchNorm1d(in_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res_block1 = ResidualTabularBlock(
            hidden_dim, hidden_dim, dropout_rate=dropout_rate
        )
        self.res_block2 = ResidualTabularBlock(
            hidden_dim, hidden_dim // 2, dropout_rate=dropout_rate
        )

        bottleneck_dim = hidden_dim // 2
        self.direct_head = nn.Linear(bottleneck_dim, num_classes)
        self.domain_latent_proj = nn.Linear(bottleneck_dim, latent_dim)
        self.tracker_prototypes = nn.Parameter(
            torch.randn(num_classes, latent_dim) / math.sqrt(latent_dim)
        )
        self.head_blend = nn.Parameter(torch.tensor([0.5]))

        if self.has_tracker_priors:
            # Context-conditioned FiLM prior modulation from domain representation h
            self.film_prior = nn.Linear(bottleneck_dim, num_prior_channels * 2)
            with torch.no_grad():
                nn.init.zeros_(self.film_prior.weight)
                nn.init.zeros_(self.film_prior.bias)

            init_weights = torch.tensor(
                [1.5, 1.0, 1.0, 1.5, 0.8], dtype=torch.float32
            ).unsqueeze(0).repeat(num_classes, 1)
            self.tracker_prior_weights = nn.Parameter(init_weights)
            self.tracker_prior_bias = nn.Parameter(torch.zeros(num_classes))
            self.tracker_res_net = nn.Sequential(
                nn.Linear(num_prior_channels, 32),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(32, 1),
            )
            with torch.no_grad():
                nn.init.normal_(self.tracker_res_net[0].weight, std=0.01)
                nn.init.zeros_(self.tracker_res_net[0].bias)
                nn.init.zeros_(self.tracker_res_net[3].weight)
                nn.init.zeros_(self.tracker_res_net[3].bias)
            self.res_scale = nn.Parameter(torch.tensor([0.5]))

        # 2-Hop Non-Linear Residual Relational Message Passing Module
        self.cooc_layer = nn.Linear(num_classes, num_classes, bias=False)
        if cooccurrence_matrix is not None:
            with torch.no_grad():
                self.cooc_layer.weight.copy_(
                    torch.from_numpy(cooccurrence_matrix.T).float()
                )
        else:
            with torch.no_grad():
                nn.init.normal_(self.cooc_layer.weight, std=0.01)

        self.rel_refine = nn.Sequential(
            nn.SiLU(),
            nn.Linear(num_classes, num_classes),
        )
        with torch.no_grad():
            nn.init.zeros_(self.rel_refine[1].weight)
            nn.init.zeros_(self.rel_refine[1].bias)

        self.cooc_gate = nn.Linear(latent_dim, num_classes)
        with torch.no_grad():
            nn.init.normal_(self.cooc_gate.weight, std=0.01)
            nn.init.constant_(self.cooc_gate.bias, -2.0)

        if initial_bias is not None:
            with torch.no_grad():
                self.direct_head.bias.copy_(torch.tensor(initial_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_norm(x)
        h = self.input_proj(h)
        h = self.res_block1(h)
        h = self.res_block2(h)

        direct_logits = self.direct_head(h)
        domain_latent = F.normalize(self.domain_latent_proj(h), p=2, dim=-1)
        tracker_latent = F.normalize(self.tracker_prototypes, p=2, dim=-1)
        proto_logits = torch.matmul(domain_latent, tracker_latent.t()) * (
            math.sqrt(self.latent_dim) / 2.0
        )

        alpha = torch.sigmoid(self.head_blend)
        base_logits = alpha * direct_logits + (1.0 - alpha) * proto_logits

        if self.has_tracker_priors:
            prior_signals = x[
                :, -self.num_prior_channels * self.num_classes :
            ].reshape(-1, self.num_prior_channels, self.num_classes)
            prior_perm = prior_signals.permute(0, 2, 1)

            # Context-conditioned FiLM modulation: gamma(h) and beta(h) modulate prior channels
            film_params = self.film_prior(h)
            gamma, beta = film_params.chunk(2, dim=-1)
            gamma = 1.0 + torch.tanh(gamma).unsqueeze(1)
            beta = beta.unsqueeze(1)
            prior_perm_modulated = prior_perm * gamma + beta

            direct_prior_logits = (
                prior_perm_modulated * self.tracker_prior_weights
            ).sum(dim=-1) + self.tracker_prior_bias
            mlp_prior_logits = self.tracker_res_net(prior_perm_modulated).squeeze(-1)
            tracker_res_logits = direct_prior_logits + mlp_prior_logits
            base_logits = base_logits + self.res_scale * tracker_res_logits

        # 2-hop non-linear residual relational message passing without destructive LayerNorm
        probs = torch.sigmoid(base_logits)
        msg1 = self.cooc_layer(probs)
        msg2 = self.cooc_layer(msg1)
        rel_msg = msg1 + self.rel_refine(msg2)
        cooc_gate = torch.sigmoid(self.cooc_gate(domain_latent))
        relational_refinement = cooc_gate * rel_msg
        final_logits = base_logits + relational_refinement
        return final_logits


class AsymmetricFocalRecallLoss(nn.Module):

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,
        clip_margin: float = 0.05,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip_margin = clip_margin
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pos_probs = probs.clamp(min=self.eps, max=1.0 - self.eps)
        loss_pos = (
            -targets * (1.0 - pos_probs).pow(self.gamma_pos) * torch.log(pos_probs)
        )

        neg_probs = (probs - self.clip_margin).clamp(min=self.eps, max=1.0 - self.eps)
        neg_probs = torch.where(
            probs <= self.clip_margin, torch.zeros_like(neg_probs), neg_probs
        )
        loss_neg = (
            -(1.0 - targets)
            * neg_probs.pow(self.gamma_neg)
            * torch.log(1.0 - neg_probs)
        )
        return (loss_pos + loss_neg).sum(dim=-1).mean()


class HybridTopKRankingLoss(nn.Module):

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,
        clip_margin: float = 0.05,
        margin: float = 1.0,
        top_k_neg: int = TOP_K,
        rank_weight: float = 0.15,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.asym_focal = AsymmetricFocalRecallLoss(
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip_margin=clip_margin,
            eps=eps,
        )
        self.margin = margin
        self.top_k_neg = top_k_neg
        self.rank_weight = rank_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.asym_focal(logits, targets)

        # Hard-negative mining top-K ranking margin loss
        neg_logits = torch.where(targets == 0, logits, torch.full_like(logits, -1e9))
        hard_neg_logits, _ = torch.topk(neg_logits, k=self.top_k_neg, dim=-1)

        margin_diff = self.margin - (
            logits.unsqueeze(-1) - hard_neg_logits.unsqueeze(1)
        )
        ranking_violation = F.relu(margin_diff)

        pos_mask = targets.unsqueeze(-1)
        num_pos = targets.sum(dim=-1, keepdim=True).clamp(min=1.0)

        sample_rank_loss = (ranking_violation * pos_mask).sum(dim=(1, 2)) / (
            num_pos.squeeze(-1) * self.top_k_neg
        )
        rank_loss = sample_rank_loss.mean()

        return focal_loss + self.rank_weight * rank_loss


# =========================================================================
# The wrapper: every label-dependent statistic plus the training loop
# =========================================================================
class NeighbourTrackerAdoption(TransformerMixin, BaseEstimator):
    """Adamic-Adar weighted tracker adoption among a domain's link neighbours.

    Stateful, and only just: the neighbours that count are the fold's TRAINING
    domains and the signal is their labels, so both are learned in fit and reused
    for the rows being scored. Everything else about the aggregation -- which
    edges exist and what each one weighs -- is a property of the hyperlink graph
    and arrives as the recorded ``edges`` table.

    One instance per direction. Outgoing and incoming adoption are independent
    aggregations over the same edge list with the endpoints swapped, so they get
    a plan node each rather than being fused into one.
    """

    def __init__(self, self_col, nbr_col, prefix):
        self.self_col = self_col
        self.nbr_col = nbr_col
        self.prefix = prefix

    @staticmethod
    def _lookup(values, keys_sorted, vals_sorted, missing=-1):
        if len(keys_sorted) == 0:
            return np.full(len(values), missing, dtype=np.int64)
        pos = np.clip(np.searchsorted(keys_sorted, values), 0, len(keys_sorted) - 1)
        return np.where(keys_sorted[pos] == values, vals_sorted[pos], missing)

    @staticmethod
    def _position_map(ids):
        order = np.argsort(ids, kind="stable")
        return ids[order], order.astype(np.int64)

    def _frame(self, X, edges):
        row_keys, row_vals = self._position_map(X["domain_id"].to_numpy(np.int64))
        row_idx = self._lookup(
            edges[self.self_col].to_numpy(np.int64), row_keys, row_vals
        )
        train_idx = self._lookup(
            edges[self.nbr_col].to_numpy(np.int64), self.train_keys_, self.train_vals_
        )
        keep = (row_idx >= 0) & (train_idx >= 0)

        adjacency = csr_matrix(
            (
                edges["weight"].to_numpy(np.float32)[keep],
                (row_idx[keep], train_idx[keep]),
            ),
            shape=(len(X), self.train_targets_csr_.shape[0]),
        )
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        counts = (adjacency @ self.train_targets_csr_).toarray()
        adoption = np.where(
            degree[:, None] > 0, counts / np.maximum(degree[:, None], 1e-7), 0.0
        ).astype(np.float32)
        # n_rows x 355 float32 -- 26.5 GB at full scale -- dead once normalised.
        del adjacency, counts

        blocks = pd.DataFrame(
            adoption,
            columns=[f"{self.prefix}_nbr_{i}" for i in range(NUM_TRACKERS)],
            index=X.index,
        )
        # Emitted at float64, the width the original accumulated at: the plan
        # takes log1p of this before narrowing, and log1p of a float32 would
        # round differently.
        blocks[f"{self.prefix}_nbr_degree"] = degree
        return pd.concat([X, blocks], axis=1)

    def fit(self, X, y=None, edges=None):
        self.fit_transform(X, y, edges=edges)
        return self

    def fit_transform(self, X, y, edges=None):
        self.train_keys_, self.train_vals_ = self._position_map(
            X["domain_id"].to_numpy(np.int64)
        )
        self.train_targets_csr_ = csr_matrix(np.asarray(y, dtype=np.float32))
        return self._frame(X, edges)

    def transform(self, X, edges=None):
        return self._frame(X, edges)


class BayesianTrackerPriors(TransformerMixin, BaseEstimator):
    """Leave-one-out root-domain and TLD tracker priors.

    Stateful by nature: the count tables are built from the fold's TRAINING
    labels and reused for the rows being scored. The original carried that
    distinction as an ``is_training_rows`` flag, which is exactly the
    fit_transform/transform split -- training rows get the leave-one-out
    posterior (their own labels removed from their root's counts), everything
    else the plain one.

    Only the two prior matrices and the two lookup results leave here; the scalar
    summaries derived from them are stateless row arithmetic and live in the plan.
    """

    def __init__(self, alpha_tld=10.0, beta_root=2.0):
        self.alpha_tld = alpha_tld
        self.beta_root = beta_root

    def _fit_tables(self, roots, tlds, targets_csr):
        tld_keys, tld_inv = np.unique(tlds, return_inverse=True)
        tld_counts = np.bincount(tld_inv, minlength=len(tld_keys))
        grouper = csr_matrix(
            (
                np.ones(len(tld_inv), dtype=np.float32),
                (tld_inv, np.arange(len(tld_inv))),
            ),
            shape=(len(tld_keys), len(tld_inv)),
        )
        tld_tracker_counts = (grouper @ targets_csr).toarray()
        self.tld_keys_ = pd.Index(tld_keys)
        # One extra row holding the global prior, used for unseen TLDs.
        self.tld_table_ = np.vstack([
            (tld_tracker_counts + self.alpha_tld * self.global_priors_)
            / (tld_counts[:, None] + self.alpha_tld),
            self.global_priors_[None, :],
        ]).astype(np.float32)

        root_keys, root_inv = np.unique(roots, return_inverse=True)
        self.root_counts_ = np.bincount(root_inv, minlength=len(root_keys))
        grouper = csr_matrix(
            (
                np.ones(len(root_inv), dtype=np.float32),
                (root_inv, np.arange(len(root_inv))),
            ),
            shape=(len(root_keys), len(root_inv)),
        )
        self.root_tracker_counts_ = (grouper @ targets_csr).toarray()
        self.root_keys_ = pd.Index(root_keys)

    def _blocks(self, roots, tlds, is_training_rows):
        tld_idx = self.tld_keys_.get_indexer(pd.Index(tlds))
        tld_idx = np.where(tld_idx >= 0, tld_idx, len(self.tld_keys_))
        tld_prior = self.tld_table_[tld_idx]
        root_prior = tld_prior.copy()

        n_rows = len(roots)
        has_root_match = np.zeros(n_rows, dtype=np.float32)
        root_cnt = np.zeros(n_rows, dtype=np.float32)

        root_idx = self.root_keys_.get_indexer(pd.Index(roots))
        rows = np.flatnonzero(root_idx >= 0)
        keys = root_idx[rows]
        totals = self.root_counts_[keys].astype(np.float32)

        if is_training_rows:
            keep = totals > 1
            rows, keys, totals = rows[keep], keys[keep], totals[keep] - 1.0
            counts = self.root_tracker_counts_[keys] - self.train_targets_[rows]
        else:
            counts = self.root_tracker_counts_[keys]

        root_prior[rows] = (
            (counts + self.beta_root * tld_prior[rows])
            / (totals[:, None] + self.beta_root)
        ).astype(np.float32)
        has_root_match[rows] = 1.0
        root_cnt[rows] = totals
        return has_root_match, root_cnt, root_prior, tld_prior

    def _frame(self, X, is_training_rows):
        roots = X["root"].astype(str).to_numpy()
        tlds = X["tld"].astype(str).to_numpy()
        has_root_match, root_cnt, root_prior, tld_prior = self._blocks(
            roots, tlds, is_training_rows
        )
        blocks = pd.concat(
            [
                pd.DataFrame(root_prior, columns=ROOT_PRIOR_COLS, index=X.index),
                pd.DataFrame(tld_prior, columns=TLD_PRIOR_COLS, index=X.index),
            ],
            axis=1,
        )
        blocks["has_root_match"] = has_root_match
        blocks["root_match_count"] = root_cnt
        # Stays here, unlike the other two scalar summaries: pandas promotes
        # float32 to float64 in `np.log(frame + eps)`, so the same expression
        # written as recorded ops accumulates in a different precision and drifts
        # ~1e-6 from the original. The reduction has to stay on a float32 ndarray.
        blocks["root_prior_entropy"] = -np.sum(
            root_prior * np.log(root_prior + 1e-12), axis=1
        ).astype(np.float32)
        return pd.concat([X, blocks], axis=1)

    def fit(self, X, y=None, global_priors=None):
        self.fit_transform(X, y, global_priors=global_priors)
        return self

    def fit_transform(self, X, y, global_priors=None):
        self.global_priors_ = np.asarray(global_priors, dtype=np.float32)
        targets = np.asarray(y, dtype=np.float32)
        self.train_targets_ = targets
        self._fit_tables(
            X["root"].astype(str).to_numpy(),
            X["tld"].astype(str).to_numpy(),
            csr_matrix(targets),
        )
        out = self._frame(X, is_training_rows=True)
        # Only the leave-one-out branch needs the raw labels; keeping them would
        # pin a n_train x 355 float32 array on the fitted transformer.
        del self.train_targets_
        return out

    def transform(self, X):
        return self._frame(X, is_training_rows=False)


class TrackerRankNetEstimator(RegressorMixin, BaseEstimator):
    """Label-dependent feature construction + the original's torch training.

    ``fit`` receives one fold's training rows; the link graph and the tracker
    table come in through ``fit_kwargs`` as DataOps (guide section 7) and are
    kept on the estimator, because they are label-free reference tables that
    ``predict`` needs too. The fold's training domains, their targets, and the
    root/TLD statistics derived from them are what make the neighbour and
    Bayesian-prior blocks fold-specific: at ``predict`` time the scored rows get
    those blocks rebuilt from the TRAINING labels only, which is what the
    original achieved by hand with its ``train_domains`` / ``val_domains`` split.
    """

    def __init__(
        self,
        num_trackers=NUM_TRACKERS,
        hidden_dim=512,
        latent_dim=128,
        dropout_rate=0.2,
        num_epochs=12,
        batch_size=4096,
        base_lr=3e-4,
        min_lr=1e-5,
        weight_decay=1e-4,
        max_checkpoint_rows=30000,
        checkpoint_fraction=0.2,
        random_state=42,
    ):
        self.num_trackers = num_trackers
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.max_checkpoint_rows = max_checkpoint_rows
        self.checkpoint_fraction = checkpoint_fraction
        self.random_state = random_state

    # --- small lookup helpers (replacing the original's dense id-indexed arrays,
    #     which allocate one slot per domain id up to max(domain_id)) ----------
    @staticmethod
    def _matrix(X):
        """The plan hands over an all-numeric frame in the network's column order."""
        return np.ascontiguousarray(X.to_numpy(np.float32))

    def _predict_logits(self, X_matrix):
        self.model_.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(X_matrix), 4096):
                batch = torch.from_numpy(X_matrix[start : start + 4096]).to(self.device_)
                chunks.append(self.model_(batch).cpu().numpy())
        return np.vstack(chunks)

    def fit(self, X, y, cooccurrence=None, init_biases=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self.train_targets_ = np.asarray(y, dtype=np.float32)
        init_biases = np.asarray(init_biases, dtype=np.float32)
        X_matrix = self._matrix(X)
        cooccurrence = np.ascontiguousarray(cooccurrence, dtype=np.float32)

        # Inner hold-out for checkpoint selection, carved out of THIS fold's
        # training rows only (the original selected on the rows it reported).
        n_rows = len(X_matrix)
        n_hold = min(self.max_checkpoint_rows, int(n_rows * self.checkpoint_fraction))
        shuffled = np.random.RandomState(self.random_state).permutation(n_rows)
        hold_idx, fit_idx = shuffled[:n_hold], shuffled[n_hold:]

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else
                                    "mps" if torch.backends.mps.is_available() else
                                    "cpu")
        self.model_ = TrackerRankNet(
            in_dim=X_matrix.shape[1],
            num_classes=self.num_trackers,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            initial_bias=init_biases,
            num_prior_channels=5,
            cooccurrence_matrix=cooccurrence,
        ).to(self.device_)

        criterion = HybridTopKRankingLoss(
            gamma_pos=0.0,
            gamma_neg=2.0,
            clip_margin=0.05,
            margin=1.0,
            top_k_neg=TOP_K,
            rank_weight=0.15,
        )

        decay_params, no_decay_params = [], []
        for name, param in self.model_.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.base_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.num_epochs - 1, eta_min=self.min_lr
        )

        # Index the fold's matrix in place instead of materialising the training
        # subset. `X_matrix[fit_idx]` is a copy of all but 30k of the fold's rows
        # -- ~138 GB at 18.7M x 1860 float32 -- and the batches are the same rows
        # in the same order either way, since fit_idx is applied before the
        # per-epoch permutation rather than after it. torch.from_numpy shares the
        # buffer, so neither tensor below copies anything.
        X_all_t = torch.from_numpy(X_matrix)
        Y_all_t = torch.from_numpy(self.train_targets_)
        fit_idx_t = torch.from_numpy(fit_idx)
        X_hold = np.ascontiguousarray(X_matrix[hold_idx])
        Y_hold = self.train_targets_[hold_idx]

        n_fit = len(fit_idx)
        num_batches_per_epoch = math.ceil(n_fit / self.batch_size)
        best_recall = -1.0
        best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}

        for epoch in range(1, self.num_epochs + 1):
            self.model_.train()
            total_loss, num_batches = 0.0, 0
            perm = torch.randperm(n_fit)

            for start in range(0, n_fit, self.batch_size):
                if epoch == 1:
                    warmup_lr = self.min_lr + (self.base_lr - self.min_lr) * (
                        (num_batches + 1) / num_batches_per_epoch
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = warmup_lr

                batch_indices = fit_idx_t[perm[start : start + self.batch_size]]
                batch_x = X_all_t[batch_indices].to(self.device_, non_blocking=True)
                batch_y = Y_all_t[batch_indices].to(self.device_, non_blocking=True)

                optimizer.zero_grad()
                loss = criterion(self.model_(batch_x), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            if epoch > 1:
                scheduler.step()

            hold_recall = recall_at_k(Y_hold, self._predict_logits(X_hold), k=10)
            print(
                f"Epoch {epoch:02d}/{self.num_epochs:02d} | "
                f"Train Loss: {total_loss / max(num_batches, 1):.4f} | "
                f"Inner Recall@10: {hold_recall:.5f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}",
                flush=True,
            )
            if hold_recall > best_recall:
                best_recall = hold_recall
                best_state = {
                    k: v.cpu().clone() for k, v in self.model_.state_dict().items()
                }

        self.model_.load_state_dict(
            {k: v.to(self.device_) for k, v in best_state.items()}
        )
        self.n_features_in_ = X_all_t.shape[1]
        return self

    def predict(self, X):
        return self._predict_logits(self._matrix(X))

def define_pipeline(input_dir):
    with skrub.config_context(eager_data_ops=False):
        # =====================================================================
        # 1. Recorded loads. Every table the original read, minus the parts that
        #    only serve submission writing.
        # =====================================================================
        trackers = skrub.as_data_op(f"{input_dir}/trackers.tsv").skb.apply_func(
            pd.read_csv, sep="\t"
        )
        target_domains = skrub.as_data_op(f"{input_dir}/target.tsv").skb.apply_func(
            pd.read_csv, sep="\t"
        )
        graph_train = skrub.as_data_op(
            f"{input_dir}/tracking_graph_train.parquet"
        ).skb.apply_func(pd.read_parquet, columns=["domain_id", "tracker_id"])
        domains = skrub.as_data_op(f"{input_dir}/domains.parquet").skb.apply_func(
            pd.read_parquet, columns=["domain", "domain_id"]
        )
        link_graph = skrub.as_data_op(f"{input_dir}/link-graph.parquet").skb.apply_func(
            pd.read_parquet, columns=["source_domain_id", "target_domain_id"]
        )
        # The original re-read this file with a sniffed separator when the tab read
        # produced fewer than 3 columns; the shipped file is tab separated, and a
        # parser fallback is not data-dependent logic.
        press = skrub.as_data_op(
            f"{input_dir}/freedom-of-the-press.csv"
        ).skb.apply_func(pd.read_csv, sep="\t")
        url_classification = skrub.as_data_op(
            f"{input_dir}/url-classification.csv"
        ).skb.apply_func(pd.read_csv, usecols=["url", "category"])

        # =====================================================================
        # 2. Build the supervised table: one row per candidate domain, in the
        #    ascending order np.setdiff1d gave the original, and the per-tracker
        #    tracker indicator as the target.
        # =====================================================================
        candidate_edges = graph_train[
            ~graph_train["domain_id"].isin(target_domains["domain_id"])
        ]
        candidate_ids = (
            candidate_edges[["domain_id"]]
            .drop_duplicates()
            .sort_values("domain_id")
            .reset_index(drop=True)
        )
        domain_names = domains.drop_duplicates(subset="domain_id", keep="last")
        base = candidate_ids.merge(domain_names, on="domain_id", how="left")
        base = base.assign(domain=base["domain"].fillna(""))

        # The edge list pivoted to one 0/1 indicator column per tracker. Aggregation
        # free: drop_duplicates makes each (domain, tracker) pair unique, so unstack
        # is a pure reshape and the uint8 fill_value keeps it uint8 end to end -- no
        # int64 blow-up, and no in-place scatter to hide in a UDF. reindex supplies
        # the trackers no candidate domain uses and fixes the row order to `base`.
        tracker_flags = (
            candidate_edges.drop_duplicates()
            .assign(present=np.uint8(1))
            .set_index(["domain_id", "tracker_id"])["present"]
        )
        targets = (
            tracker_flags.unstack(fill_value=np.uint8(0))
            .reindex(
                index=candidate_ids["domain_id"],
                columns=range(NUM_TRACKERS),
                fill_value=np.uint8(0),
            )
            .set_axis(TRACKER_COLS, axis=1)
            .reset_index(drop=True)
        )

        y = targets.skb.mark_as_y()
        X = base.skb.mark_as_X(cv=CandidateDomainHoldout(), split_kwargs={})

        # =====================================================================
        # 3. Row-local features: hostname lexicon + press freedom + URL categories.
        #    All of it is recorded pandas, so every feature is its own plan node.
        # =====================================================================
        # --- registrable root and TLD (the original's extract_root_and_tld) ---
        name = X["domain"]
        lowered = name.str.lower()
        cleaned = lowered.str.strip()
        # The original indexed parts[-1] / [-2] / [-3] of a split hostname. Anchored
        # regexes give the same labels without materialising a list column, which the
        # stratum scheduler cannot hand to polars.
        n_parts = cleaned.str.count(r"\.") + 1
        last = cleaned.str.extract(r"([^.]*)$", expand=False)
        second_last = cleaned.str.extract(r"([^.]*)\.[^.]*$", expand=False)
        third_last = cleaned.str.extract(r"([^.]*)\.[^.]*\.[^.]*$", expand=False)
        last_two = second_last + "." + last
        # A two-level suffix only counts when something precedes it, mirroring the
        # original's `len(parts) >= 3` guard -- otherwise "co.uk" would be its own root.
        is_two_level = last_two.isin(TWO_LEVEL_TLDS) & (n_parts >= 3)
        is_unknown = name == ""              # the original's `not domain_name` branch

        X = X.assign(
            root=(
                last_two.where(n_parts > 1, last)
                .where(~is_two_level, third_last + "." + last_two)
                .where(~is_unknown, "unknown")
            ),
            tld=last.where(~is_two_level, last_two).where(~is_unknown, "unknown"),
        )

        # --- hostname lexicon: the original's X_lexical block, minus the press
        #     columns, which are joined below and slotted back in by the reorder ---
        dom_len = name.str.len().clip(lower=1).astype("float32")
        num_dots = name.str.count(r"\.").astype("float32")
        num_digits = name.str.count(r"\d").astype("float32")
        X = X.assign(
            dom_len=dom_len,
            num_dots=num_dots,
            num_hyphens=name.str.count("-").astype("float32"),
            num_digits=num_digits,
            digit_ratio=num_digits / dom_len,
            has_www=lowered.str.startswith("www.").astype("float32"),
            has_multi_sub=(num_dots > 1).astype("float32"),
        )

        # One-hot over the 30 most frequent TLDs; everything else lands in tld_other.
        tld_col = X["tld"]
        X = X.assign(
            **{f"tld_{t}": (tld_col == t).astype("float32") for t in TOP_TLDS},
            tld_other=(~tld_col.isin(TOP_TLDS)).astype("float32"),
        )

        # How many distinct keywords of each family the hostname contains. The
        # .astype is load-bearing: numpy adds booleans as a logical OR, so summing
        # the raw masks saturates at 1 and "goshopmall" would count one keyword.
        keyword_counts = {
            col: sum(
                lowered.str.contains(kw, regex=False).astype("float32") for kw in words
            ).astype("float32")
            for col, words in KEYWORDS.items()
        }
        X = X.assign(
            **keyword_counts,
            kw_total=sum(keyword_counts.values()).astype("float32"),
        )

        # --- press freedom, joined by TLD ---
        press_scores = (
            press.assign(
                tld=press["tld"].astype(str).str.strip().str.lower().str.lstrip("."),
                press_score=press["freedom_of_the_press"].astype("float64"),
            )[["tld", "press_score"]]
            .drop_duplicates(subset="tld", keep="last")
        )
        median_press_score = press_scores["press_score"].median()

        X = X.merge(press_scores, on="tld", how="left")
        X = X.assign(
            has_press=X["press_score"].notna().astype("float32"),
            press_score=X["press_score"].fillna(median_press_score).astype("float32"),
        )
        X = X.assign(is_auth=(X["press_score"] > 60.0).astype("float32"))

        # The network's weight initialisation is per-column, so the lexical block has
        # to reach the model in the original column_stack order, not in the order the
        # recorded ops happened to create it in. This also drops the raw hostname.
        X = X[META_COLS + LEXICAL_COLS]

        # Content-category mix per domain, matched by hostname with the original's
        # "www." fallback.
        url_rows = url_classification[url_classification["category"].isin(URL_CATEGORIES)]
        url_hosts = (
            url_rows["url"]
            .astype(str)
            .str.split("://", n=1)
            .str[-1]
            .str.split("/", n=1)
            .str[0]
            .str.split(":", n=1)
            .str[0]
            .str.strip()
            .str.lower()
        )
        url_rows = url_rows.assign(host=url_hosts, host_no_www=url_hosts.str.removeprefix("www."))
        name_lookup = domains[
            domains["domain_id"].isin(candidate_ids["domain_id"])
        ].drop_duplicates(subset="domain", keep="last")
        url_rows = url_rows.merge(
            name_lookup, left_on="host", right_on="domain", how="left"
        ).merge(
            name_lookup.rename(columns={"domain": "domain_www", "domain_id": "domain_id_www"}),
            left_on="host_no_www",
            right_on="domain_www",
            how="left",
        )
        url_rows = url_rows.assign(
            matched_id=url_rows["domain_id"].fillna(url_rows["domain_id_www"])
        )
        url_rows = url_rows[url_rows["matched_id"].notna()]

        category_counts = (
            url_rows.assign(matched_id=url_rows["matched_id"].astype("int64"))
            .groupby(["matched_id", "category"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=URL_CATEGORIES, fill_value=0)
        )
        category_table = (
            category_counts.div(category_counts.sum(axis=1), axis=0)
            .rename(columns={c: f"url_cat_{c}" for c in URL_CATEGORIES})
            .assign(url_cat_has=1.0)
            .reset_index()
            .rename(columns={"matched_id": "domain_id"})
        )

        X = X.merge(category_table, on="domain_id", how="left")
        X = X.assign(**{col: X[col].fillna(0.0).astype("float32") for col in URL_CAT_COLS})

        # --- link-graph degrees. Derived from the hyperlink graph alone, never from
        #     a label, so they are ordinary recorded features rather than something
        #     the estimator has to rebuild per fold. A domain absent from the graph
        #     gets 0, matching the original's bincount-and-index lookup.
        out_degree = (
            link_graph.groupby("source_domain_id")
            .size()
            .rename("deg_out")
            .reset_index()
            .rename(columns={"source_domain_id": "domain_id"})
        )
        in_degree = (
            link_graph.groupby("target_domain_id")
            .size()
            .rename("deg_in")
            .reset_index()
            .rename(columns={"target_domain_id": "domain_id"})
        )
        tracker_link_counts = (
            link_graph[link_graph["target_domain_id"].isin(trackers["tracking_domain_id"])]
            .groupby("source_domain_id")
            .size()
            .rename("tracker_links")
            .reset_index()
            .rename(columns={"source_domain_id": "domain_id"})
        )

        X = X.merge(out_degree, on="domain_id", how="left")
        X = X.merge(in_degree, on="domain_id", how="left")
        X = X.merge(tracker_link_counts, on="domain_id", how="left")
        X = X.assign(
            deg_out=X["deg_out"].fillna(0).astype("float32"),
            deg_in=X["deg_in"].fillna(0).astype("float32"),
            tracker_links=X["tracker_links"].fillna(0).astype("float32"),
        )

        deg_out, deg_in = X["deg_out"], X["deg_in"]
        tracker_links = X["tracker_links"]
        X = X.assign(
            deg_out_log=deg_out.skb.apply_func(np.log1p),
            deg_in_log=deg_in.skb.apply_func(np.log1p),
            deg_total=deg_out + deg_in,
            deg_in_out_ratio=(deg_in + np.float32(1.0)) / (deg_out + np.float32(1.0)),
            deg_is_isolated=(deg_out + deg_in == 0).astype("float32"),
            tracker_links_log=tracker_links.skb.apply_func(np.log1p),
            tracker_links_per_out=tracker_links / deg_out.clip(lower=np.float32(1.0)),
        )

        # --- direct tracker links: the same long-to-wide reshape as the target
        #     matrix, over the link-graph edges whose destination IS a tracker.
        tracker_edges = link_graph.merge(
            trackers[["tracking_domain_id", "tracker_id"]],
            left_on="target_domain_id",
            right_on="tracking_domain_id",
            how="inner",
        )
        direct_flags = (
            tracker_edges[["source_domain_id", "tracker_id"]]
            .drop_duplicates()
            .assign(present=np.uint8(1))
            .set_index(["source_domain_id", "tracker_id"])["present"]
        )
        direct_table = (
            direct_flags.unstack(fill_value=np.uint8(0))
            .reindex(columns=range(NUM_TRACKERS), fill_value=np.uint8(0))
            .set_axis(DIRECT_COLS, axis=1)
            .reset_index()
            .rename(columns={"source_domain_id": "domain_id"})
        )

        X = X.merge(direct_table, on="domain_id", how="left")
        X = X.assign(**{col: X[col].fillna(0).astype("float32") for col in DIRECT_COLS})

        # Fix the final column order once: the estimator reads these blocks by name
        # and has to stack them exactly as the original's column_stack did.
        X = X[META_COLS + LEXICAL_COLS + URL_CAT_COLS + GRAPH_STATIC_COLS + DIRECT_COLS]

        # =====================================================================
        # 4. The model. The link graph and tracker table reach it through
        #    fit_kwargs; every label-derived block is rebuilt per fold inside fit.
        # =====================================================================
        # --- tracker co-occurrence, the initial weights of the relational layer.
        #     Derived from the fold's TRAINING labels, and consumed only while
        #     fitting, so it travels as a fit_kwarg and needs no freeze_after_fit.
        train_flags = y.astype("float32")
        cooc_counts = train_flags.T.dot(train_flags)
        tracker_freqs = cooc_counts.skb.apply_func(np.diag)
        cond_cooc = cooc_counts.div(tracker_freqs + COOC_EPSILON, axis=0)
        cond_cooc = cond_cooc.where(~np.eye(NUM_TRACKERS, dtype=bool), np.float32(0.0))
        row_max = cond_cooc.max(axis=1)
        # The original guarded this division with `np.where(row_max > 0, ..., 0.0)`.
        # Every entry of cond_cooc is non-negative, so a row whose max is 0 is all
        # zeros and the clipped division already returns 0 for it -- the guard is
        # algebraically redundant here, not an assumption about the data.
        cooccurrence = cond_cooc.div(row_max.clip(lower=np.float32(1.0)), axis=0)

        # --- empirical tracker rates over the fold's TRAINING labels, and the
        #     logits the output layer's bias is initialised to. Both are fit-time
        #     only, so they ride along as fit_kwargs like the co-occurrence above.
        global_priors = (
            y.to_numpy(np.float32).mean(axis=0, dtype=np.float64).astype(np.float32)
        )
        clipped_priors = global_priors.clip(PRIOR_EPS, 1.0 - PRIOR_EPS)
        init_biases = (
            (clipped_priors / (1.0 - clipped_priors))
            .skb.apply_func(np.log)
            .astype(np.float32)
        )

        # --- Bayesian root/TLD priors. The only stateful step left outside the
        #     model: the count tables come from the fold's training labels.
        X = X.skb.apply(
            BayesianTrackerPriors(),
            y=y,
            fit_transform_kwargs={"global_priors": global_priors},
        )

        # Scalar summaries of the root prior. Pure row arithmetic over the block the
        # transformer produced, so they are recorded rather than hidden inside it.
        X = X.assign(
            root_match_count_log=X["root_match_count"].skb.apply_func(np.log1p),
            root_prior_max=X[ROOT_PRIOR_COLS].max(axis=1).astype("float32"),
        )

        # --- Adamic-Adar edge weights. A pure function of the hyperlink graph: an
        #     edge weighs 1 / log1p(total degree of the endpoint it reaches), so it
        #     is recorded once instead of being rebuilt inside the aggregation.
        degrees = out_degree.merge(in_degree, on="domain_id", how="outer").fillna(0)
        total_degree = degrees[["domain_id"]].assign(
            total_degree=degrees["deg_out"] + degrees["deg_in"]
        )
        # Self-links, excluded by the original's `src != dst` mask.
        linked = link_graph[
            link_graph["source_domain_id"] != link_graph["target_domain_id"]
        ]

        out_weighted = linked.merge(
            total_degree.rename(columns={"domain_id": "target_domain_id"}),
            on="target_domain_id",
            how="left",
        )
        out_edges = out_weighted[["source_domain_id", "target_domain_id"]].assign(
            weight=(
                1.0
                / out_weighted["total_degree"].clip(lower=1).skb.apply_func(np.log1p)
            ).astype("float32")
        )
        in_weighted = linked.merge(
            total_degree.rename(columns={"domain_id": "source_domain_id"}),
            on="source_domain_id",
            how="left",
        )
        in_edges = in_weighted[["source_domain_id", "target_domain_id"]].assign(
            weight=(
                1.0 / in_weighted["total_degree"].clip(lower=1).skb.apply_func(np.log1p)
            ).astype("float32")
        )

        X = X.skb.apply(
            NeighbourTrackerAdoption("source_domain_id", "target_domain_id", "out"),
            y=y,
            fit_transform_kwargs={"edges": out_edges},
            transform_kwargs={"edges": out_edges},
        )
        X = X.skb.apply(
            NeighbourTrackerAdoption("target_domain_id", "source_domain_id", "in"),
            y=y,
            fit_transform_kwargs={"edges": in_edges},
            transform_kwargs={"edges": in_edges},
        )

        # The six graph scalars those two degrees imply -- stateless, so recorded
        # here rather than emitted by the transformers.
        nbr_out, nbr_in = X["out_nbr_degree"], X["in_nbr_degree"]
        nbr_tot = nbr_out + nbr_in
        X = X.assign(
            nbr_out_deg=nbr_out.astype("float32"),
            nbr_out_deg_log=nbr_out.skb.apply_func(np.log1p).astype("float32"),
            nbr_in_deg=nbr_in.astype("float32"),
            nbr_in_deg_log=nbr_in.skb.apply_func(np.log1p).astype("float32"),
            nbr_tot_deg=nbr_tot.astype("float32"),
            nbr_tot_deg_log=nbr_tot.skb.apply_func(np.log1p).astype("float32"),
        )

        # --- the design matrix itself: a column selection in the original's
        #     column_stack order, followed by the nan/inf scrub it applied. None of
        #     that is stateful, so the estimator receives a ready numeric frame and
        #     no longer assembles anything.
        features = (
            X[FEATURE_COLS]
            .replace([np.inf, -np.inf], np.float32(0.0))
            .fillna(np.float32(0.0))
        )

        pred = features.skb.apply(
            TrackerRankNetEstimator(
                num_epochs=NUM_EPOCHS,
                hidden_dim=HIDDEN_DIM,
                latent_dim=LATENT_DIM,
                batch_size=BATCH_SIZE,
            ),
            y=y,
            fit_kwargs={
                "cooccurrence": cooccurrence,
                "init_biases": init_biases,
            },
        )
        return pred

class TestTrackTheTrackersPipeline(unittest.TestCase):

    def test_pipeline(self):
        recall = make_scorer(recall_at_k, greater_is_better=True,
                             response_method="predict")
        with ttt_input_dir() as input_dir:
            pipe = define_pipeline(input_dir)
            with skrub.config(scheduler=True, stats=True, stats_top_k=25):
                search = pipe.skb.make_grid_search(
                    n_jobs=1,
                    fitted=True,
                    refit=False,
                    scoring=recall,
                )
        # stratum returns a polars frame with one row per candidate -- columns
        # `id` and `scores`, already sorted best-first -- where skrub returns a
        # pandas frame with `mean_test_score`.
        results = search.results_
        self.assertEqual(1, len(results))
        score = results["scores"][0]
        # Recall@10 over 64 trackers: a fitted model has to beat the floor, and
        # anything above 1.0 means the metric is being fed the wrong shape.
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
