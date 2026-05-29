from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from utils import *
from predictors import *
from Pareto_fn import pareto_fn
from pcgrad_fn import pcgrad_fn


def _nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2, eps: float = 1e-12) -> torch.Tensor:
    """NT-Xent (SimCLR) loss.

    Args:
        z1, z2: [N, D] embeddings from two stochastic views of the same samples.
        tau: temperature.

    Returns:
        scalar loss.
    """
    if z1 is None or z2 is None:
        return torch.tensor(0.0, device=z1.device if z1 is not None else (z2.device if z2 is not None else 'cpu'))

    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError(f"NT-Xent expects 2D tensors, got z1={tuple(z1.shape)} z2={tuple(z2.shape)}")
    if z1.shape[0] != z2.shape[0]:
        n = min(z1.shape[0], z2.shape[0])
        z1 = z1[:n]
        z2 = z2[:n]
    n = z1.shape[0]
    if n <= 1:
        # no meaningful contrast when batch has <=1 samples
        return torch.zeros((), device=z1.device)

    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    z = torch.cat([z1, z2], dim=0)  # [2N, D]

    sim = (z @ z.t()) / max(float(tau), float(eps))  # [2N, 2N]
    # remove self-similarity from denominator
    mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float('-inf'))

    # positive pairs: (i, i+N) and (i+N, i)
    pos_idx = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=False)
    loss = -log_prob[torch.arange(2 * n, device=z.device), pos_idx]
    return loss.mean()


def _subsample_for_contrast(z1: torch.Tensor, z2: torch.Tensor, max_samples: int, seed: int = 0):
    """Subsample paired embeddings to avoid O(N^2) memory blowup in NT-Xent.

    Keeps aligned pairs (same indices for z1 and z2).
    """
    if max_samples is None:
        return z1, z2
    m = int(max_samples)
    if m <= 0:
        return z1, z2
    n = int(min(z1.shape[0], z2.shape[0]))
    if n <= m:
        return z1[:n], z2[:n]

    # deterministic subsample
    g = torch.Generator(device='cpu')
    g.manual_seed(int(seed))
    idx = torch.randperm(n, generator=g)[:m]
    return z1[idx.to(z1.device)], z2[idx.to(z2.device)]


# threshold adjusting for target recall (pick point just before recall drops below target)
def get_target_recall_threshold(labels, probs, target_recall: float):
    """Return (pos_f1, thres, cm) chosen by a target-recall rule.

        Strategy (robust version of the snippet you referenced):
        - Compute precision-recall curve.
        - Align candidates to (prec[:-1], rec[:-1], ths).
        - Sort candidates by recall descending (so recall is guaranteed to go from high->low).
        - Scan; when recall first drops below `target_recall`, choose the previous point (i-1),
            i.e., the last point that still meets the target.

    Notes:
    - sklearn's `precision_recall_curve` returns `prec`, `rec` of length T+1 and `ths` of length T.
      We align candidates to `prec[:-1]`, `rec[:-1]` with thresholds `ths`.
    - If recall never drops below target, we fall back to the last threshold.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if labels.size == 0:
        best_cm = confusion_matrix(labels, np.zeros_like(labels), labels=[0, 1])
        return 0.0, 0.5, best_cm

    target_recall = float(np.clip(target_recall, 0.0, 1.0))

    prec, rec, ths = precision_recall_curve(labels, probs)
    if ths.size == 0:
        # degenerate case (e.g., probs constant)
        th = 0.5
        _, _, f1, cm = _prf_at_threshold(labels, probs, th)
        return float(f1), float(th), cm

    prec_t = prec[:-1]
    rec_t = rec[:-1]

    # Enforce the "recall goes from high to low" scanning order.
    order = np.argsort(-rec_t, kind="mergesort")
    rec_s = rec_t[order]
    prec_s = prec_t[order]
    ths_s = ths[order]

    # find first index where recall drops below target
    best_i = None
    for i in range(rec_s.shape[0]):
        if rec_s[i] < target_recall:
            best_i = max(0, i - 1)
            break
    if best_i is None:
        best_i = int(ths_s.shape[0] - 1)

    best_thre = float(ths_s[best_i])
    p, r, f1, cm = _prf_at_threshold(labels, probs, best_thre)
    return float(f1), float(best_thre), cm


# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        # preds[probs[:,1] > thres] = 1
        preds[probs > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


def get_best_f1_and_cm(labels, probs):
    """Return (best_pos_f1, best_thre, cm) where the selection metric is POSITIVE-class F1.

    We choose threshold by maximizing the POSITIVE-class F1 score along the precision-recall curve.
    This matches a typical "best-F1" thresholding strategy.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if labels.size == 0:
        best_cm = confusion_matrix(labels, np.zeros_like(labels), labels=[0, 1])
        return 0.0, 0.5, best_cm

    prec, rec, ths = precision_recall_curve(labels, probs)
    # ths has length len(prec)-1; align to (prec[:-1], rec[:-1])
    prec_t = prec[:-1]
    rec_t = rec[:-1]

    f1_t = 2.0 * prec_t * rec_t / (prec_t + rec_t + 1e-9)
    if f1_t.size == 0:
        best_thre = 0.5
    else:
        best_i = int(np.nanargmax(f1_t))
        best_thre = float(ths[best_i])

    p, r, f1, cm = _prf_at_threshold(labels, probs, best_thre)
    return float(f1), float(best_thre), cm


def _prf_at_threshold(labels, probs, thres: float):
    """Compute precision/recall/f1 (binary, positive label=1) at given threshold."""
    # Use ">=" to be robust to heavily-tied probability outputs.
    # With strict ">", fixed-FPR thresholds can degenerate to FP=0 when many negatives equal the threshold.
    preds = (probs >= thres).astype(np.int64)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, cm


def _prf_at_fpr(labels: np.ndarray, probs: np.ndarray, target_fpr: float):
    """Compute PRF1+CM at a threshold chosen to match a target FPR.

    We pick threshold so that approximately `target_fpr` fraction of NEGATIVE samples
    are predicted positive (i.e., FP/(FP+TN) ~= target_fpr).
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if labels.size == 0:
        return 0.0, 0.0, 0.0, 0.5, confusion_matrix(labels, np.zeros_like(labels), labels=[0, 1])

    neg_scores = probs[labels == 0]
    if neg_scores.size == 0:
        # no negatives => FPR undefined; fall back to a high threshold
        th = float(np.max(probs) + 1e-6)
        p, r, f1, cm = _prf_at_threshold(labels, probs, th)
        return p, r, f1, th, cm

    target_fpr = float(np.clip(target_fpr, 0.0, 1.0))

    # Choose threshold by negative-score quantile.
    # If we predict positive when prob > th, then among negatives we want:
    #   P(neg_prob > th) ~= target_fpr
    # => th ~= quantile(neg_prob, 1 - target_fpr)
    q = 1.0 - target_fpr
    if q <= 0.0:
        th = float(np.min(neg_scores) - 1e-12)
    elif q >= 1.0:
        th = float(np.max(neg_scores) + 1e-12)
    else:
        th = float(np.quantile(neg_scores, q, method="linear"))

    p, r, f1, cm = _prf_at_threshold(labels, probs, th)
    return p, r, f1, th, cm


def get_quantile_thresholds(probs, n: int = 99):
    """Adaptive threshold candidates built from prob quantiles.

    Rationale: when probs are tightly clustered (e.g., 0.468~0.471), a coarse grid like
    [0.05, 0.10, ..., 0.95] can't place candidates inside the real score range.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.size == 0:
        return np.linspace(0.05, 0.95, 19)

    qs = np.linspace(0.01, 0.99, int(n))
    ths = np.quantile(probs, qs)
    ths = np.clip(ths, 0.0, 1.0)
    ths = np.unique(ths)

    # If probs are (nearly) constant, build a tiny neighborhood around that value.
    if ths.size <= 1:
        v = float(ths[0]) if ths.size == 1 else float(np.mean(probs))
        ths = np.unique(np.clip(np.array([v - 1e-3, v - 5e-4, v, v + 5e-4, v + 1e-3]), 0.0, 1.0))

    return ths


def get_topk_thresholds(labels, probs, k: int = 5):
    """Scan thresholds and return top-k by POSITIVE-class F1 with PRF1@thres and CM."""
    rows = []
    thresholds = get_quantile_thresholds(probs, n=99)
    for thres in thresholds:
        p, r, f1, cm = _prf_at_threshold(labels, probs, float(thres))
        rows.append({
            'th': float(thres),
            'p_pos': float(p),
            'r_pos': float(r),
            'f1_pos': float(f1),
            'cm': cm,
        })
    rows.sort(key=lambda x: (x['f1_pos'], x['r_pos'], x['p_pos'], -x['th']), reverse=True)
    return rows[: max(1, int(k))]


def get_fixed_threshold_points():
    """A small set of thresholds to always print for debugging."""
    return [0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]

LABEL_DICT_KEYS = {
    'n':"node_labels",
    'e':'edge_labels',
    'g':'graph_labels',
}

class UnifyMLPDetector(object):
    def __init__(self, pretrain_model, dataset, dataloaders, cross_mode, args):
        self.args = args

        # optional: persist debug output to file for later inspection
        self._debug_log_fp = None
        if getattr(self.args, 'log_debug', False):
            try:
                import os
                from datetime import datetime

                os.makedirs('../results/debug_logs', exist_ok=True)
                ts = datetime.now().strftime('%Y%m%d-%H%M%S')
                ds_name = getattr(dataset, 'name', 'dataset')
                safe_ds = str(ds_name).replace('/', '_').replace('\\', '_')
                safe_cm = str(cross_mode).replace('/', '_').replace('\\', '_')
                log_path = f"../results/debug_logs/{safe_ds}.{safe_cm}.{ts}.log"
                self._debug_log_fp = open(log_path, 'a', encoding='utf-8')
                self._debug_log_fp.write(f"# debug log: dataset={ds_name} cross_mode={cross_mode} ts={ts}\n")
                self._debug_log_fp.flush()
                print(f"[log_debug] saving debug logs to: {log_path}")
            except Exception as e:
                self._debug_log_fp = None
                print(f"[log_debug] failed to initialize debug log file: {e}")

        self.train_dataloader = dataloaders[0]
        self.val_dataloader = dataloaders[1]
        self.test_dataloader = dataloaders[2]

        # the loss route
        input_route, output_route = cross_mode.split('2')
        self.input_route = [c for c in input_route] # ['n', 'e', 'g']
        self.output_route = [c for c in output_route] # ['n', 'e', 'g'] # the output of the model

        self.model = UNIMLP_E2E(
            in_feats=pretrain_model.in_dim,
            embed_dims=pretrain_model.embed_dim,
            khop=args.khop,
            activation=args.act_ft,
            graph_batch_num=args.batch_size,
            stitch_mlp_layers=args.stitch_mlp_layers,
            final_mlp_layers=args.final_mlp_layers,
            pretrain_model=pretrain_model,
            output_route=output_route,
            input_route=input_route,
            dropout_rate=args.dropout
        ).to(args.device)

    # (Rayleigh(A) thresholding removed)

        self.loss_weight_dict = {}
        if 'n' in self.output_route:
            node_ab_count, node_total_count = sum([x.sum() for x in dataset.node_label]), sum(x.shape[0] for x in dataset.node_label)
            self.loss_weight_dict['n'] = (
                1/(node_ab_count/node_total_count),
                args.node_loss_weight 
            )
        if 'e' in self.output_route:
            edge_ab_count, edge_total_count = sum([x.sum() for x in dataset.edge_label]), sum(x.shape[0] for x in dataset.edge_label)
            self.loss_weight_dict['e'] = (
                1/(edge_ab_count/edge_total_count),
                args.edge_loss_weight 
            )
        if 'g' in self.output_route:
            graph_ab_count, graph_total_count = dataset.graph_label.sum(), dataset.graph_label.shape[0]
            self.loss_weight_dict['g'] = (
                1/(graph_ab_count/graph_total_count),
                args.graph_loss_weight 
            )

        # masks for single graph
        if dataset.is_single_graph:
            mask_dicts = {}
            self.is_single_graph = True
            if 'n' in self.output_route:
                mask_dicts['n'] = {
                    'train': dataset.train_mask_node_cur,
                    'val': dataset.val_mask_node_cur,
                    'test': dataset.test_mask_node_cur
                }
            if 'e' in self.output_route:
                mask_dicts['e'] = {
                    'train': dataset.train_mask_edge_cur,
                    'val': dataset.val_mask_edge_cur,
                    'test': dataset.test_mask_edge_cur
                }
            if 'g' in self.output_route:
                # single graph cannot be classified
                raise NotImplementedError
            
            self.model.mask_dicts = mask_dicts
            self.model.single_graph = True

        self.best_score = -1
        self.patience_knt = 0


    def _emit_debug(self, msg: str):
        """Print + optionally append to debug log file."""
        print(msg)
        if self._debug_log_fp is not None:
            try:
                self._debug_log_fp.write(msg + "\n")
            except Exception:
                pass


    def _emit_epoch_summary(self, epoch: int, split: str, score_dict: dict):
        """Emit one-line summary for trend tracking."""
        if not getattr(self.args, 'print_cm', False):
            return
        every = int(getattr(self.args, 'debug_summary_every', 0) or 0)
        if every <= 0:
            return
        if epoch % every != 0:
            return
        for k in self.output_route:
            sd = score_dict.get(k, {})
            cm = sd.get('CM', None)
            th = sd.get('BestThres', None)
            if cm is None or th is None:
                continue
            try:
                tn, fp, fn, tp = np.asarray(cm).ravel().tolist()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            except Exception:
                continue
            msg = (
                f"[Summary] epoch={epoch} split={split} task={NAME_MAP[k]} "
                f"best_th={float(th):.6f} P={precision:.4f} R={recall:.4f} F1={f1:.4f} "
                f"TP={tp} FP={fp} TN={tn} FN={fn} FPR={fpr:.4f}"
            )
            self._emit_debug(msg)


    def _run_split_eval(self, dataloader, scen: str):
        """Run model on a dataloader and return (labels_dict_mul, probs_dict_mul, loss_items_total).

        Notes:
        - labels/probs are concatenated 1D tensors per route ('n'/'e'/'g').
        - This mirrors the existing val/test eval loops so we can reuse it at the end.
        """
        labels_dict_mul = {k: [] for k in self.output_route}
        probs_dict_mul = {k: [] for k in self.output_route}
        loss_items_total = {k: 0 for k in self.output_route}

        for batched_data in dataloader:
            batched_graph, batched_labels_dict, batched_khop_graph = batched_data
            batched_graph = batched_graph.to(self.args.device)
            for k, v in batched_labels_dict.items():
                batched_labels_dict[k] = v.to(self.args.device)
                if k[0] in self.output_route:
                    labels_dict_mul[k[0]].append(v)
            batched_khop_graph = batched_khop_graph.to(self.args.device)

            self.model.eval()
            with torch.no_grad():
                logits_dict = self.model(
                    batched_graph,
                    batched_graph.ndata['feature'],
                    batched_khop_graph,
                    scen=scen,
                )
                _, loss_items = self.get_loss(logits_dict, labels_dict=batched_labels_dict)
                for k in self.output_route:
                    loss_items_total[k] += loss_items[k]

                probs = self.get_probs(logits_dict)
                for k in probs:
                    probs_dict_mul[k].append(probs[k])

            del batched_data
            del batched_graph
            del batched_labels_dict
            del batched_khop_graph
            del logits_dict
            del probs

        for k in self.output_route:
            labels_dict_mul[k] = torch.cat([t for t in labels_dict_mul[k]])
            probs_dict_mul[k] = torch.cat([t for t in probs_dict_mul[k]])

        return labels_dict_mul, probs_dict_mul, loss_items_total
        


    def get_loss(self, logits_dict={}, labels_dict={}):
        loss_items_dict = {'n': 0, 'e': 0, 'g': 0}
        loss = None

        loss_list = []
        w_list = []
        c_list = []

        for o_r in logits_dict:
            partial_loss = F.cross_entropy(logits_dict[o_r], labels_dict[LABEL_DICT_KEYS[o_r]], weight=torch.tensor([1., self.loss_weight_dict[o_r][0]], device=self.args.device))
            if o_r in self.input_route:
                # loss = partial_loss if loss is None else (loss + partial_loss * self.loss_weight_dict[o_r][1])
                loss_list.append(partial_loss)
                w_list.append(1.0/len(self.input_route)) # FIXME: default loss average
                c_list.append(0.01)
            loss_items_dict[o_r] = partial_loss.item()

        # return loss_list, loss_items_dict
        
        new_w_list = pareto_fn(w_list, c_list, model=self.model, num_tasks=len(loss_list), loss_list=loss_list)
        loss = 0
        for i in range(len(w_list)):
            loss += new_w_list[i]*loss_list[i]
        
        return loss, loss_items_dict


    def _final_test_report(self, score_test: dict):
        """Always emit final test P/R/F1/CM and persist to debug log (if enabled)."""
        if not getattr(self.args, 'print_cm', False):
            return
        for k in self.output_route:
            sd = score_test.get(k, {})
            cm = sd.get('CM', None)
            th = sd.get('BestThres', None)
            if cm is None or th is None:
                continue
            try:
                tn, fp, fn, tp = np.asarray(cm).ravel().tolist()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            except Exception:
                continue
            msg = (
                f"[FinalTest] task={NAME_MAP[k]} best_th={float(th):.6f} "
                f"P={precision:.4f} R={recall:.4f} F1={f1:.4f} "
                f"TP={tp} FP={fp} TN={tn} FN={fn} FPR={fpr:.4f} cm={np.asarray(cm).tolist()}"
            )
            self._emit_debug(msg)

            # Also print stage breakdown for misclassified EDGE samples when available.
            if k == 'e':
                try:
                    self._print_misclassified_edge_stages(best_th=float(th), max_print=int(getattr(self.args, 'mis_stage_max_print', 30)))
                except Exception as ex:
                    self._emit_debug(f"[MisStage][warn] failed to print misclassified edge stages: {ex}")


    def _print_misclassified_edge_stages(self, best_th: float, max_print: int = 30):
        """Print Stage info for FP/FN edges on the TEST split.

        Requires graphs to carry `edata['stage_id']` (added by build_flow_node_graph.py).
        """
        # First, gather y/prob in the canonical order.
        labels_dict_test_mul, probs_dict_test_mul, _ = self._run_split_eval(self.test_dataloader, scen='test')
        y_t = labels_dict_test_mul.get('e', None)
        p_t = probs_dict_test_mul.get('e', None)
        if y_t is None or p_t is None:
            return
        y = y_t.detach().cpu().numpy().reshape(-1).astype(np.int64)
        p = p_t.detach().cpu().numpy().reshape(-1).astype(np.float64)
        preds = (p >= float(best_th)).astype(np.int64)

        # Then, gather stage_id in the SAME concatenation order as dataloader iteration.
        stage_ids_chunks = []
        stage_id2name = None

        # Try to load mapping from graph file-level label_dict (most reliable across DGL versions)
        if stage_id2name is None:
            try:
                from dgl.data.utils import load_graphs

                ds = getattr(self, 'dataset', None)
                ds_name = None
                ds_prefix = None
                if ds is not None:
                    ds_name = getattr(ds, 'name', None)
                    ds_prefix = getattr(ds, 'prefix', None)
                if ds_prefix is not None and ds_name is not None:
                    _, lbl = load_graphs(str(ds_prefix) + str(ds_name))
                    b = lbl.get('stage_id2name__bytes', None)
                    off = lbl.get('stage_id2name__offsets', None)
                    if b is not None and off is not None:
                        b = np.asarray(b.detach().cpu().numpy(), dtype=np.uint8).tobytes()
                        off = np.asarray(off.detach().cpu().numpy(), dtype=np.int64)
                        names = []
                        for i in range(int(off.shape[0] - 1)):
                            s = b[int(off[i]) : int(off[i + 1])]
                            names.append(s.decode('utf-8', errors='replace'))
                        stage_id2name = names
            except Exception:
                stage_id2name = None
        for batched_data in self.test_dataloader:
            batched_graph, _, _ = batched_data
            if 'stage_id' in batched_graph.edata:
                stage_ids_chunks.append(batched_graph.edata['stage_id'].detach().cpu().reshape(-1))
            else:
                stage_ids_chunks.append(torch.full((batched_graph.num_edges(),), -1, dtype=torch.int64))

            # try to read mapping (if saved)
            if stage_id2name is None:
                try:
                    stage_id2name = getattr(batched_graph, 'graph_data', {}).get('stage_id2name', None)
                    if stage_id2name is None:
                        stage_id2name = getattr(batched_graph, 'gdata', {}).get('stage_id2name', None)
                except Exception:
                    stage_id2name = None

        stage_ids = torch.cat(stage_ids_chunks, dim=0).numpy().astype(np.int64)
        if stage_ids.shape[0] != y.shape[0]:
            self._emit_debug(f"[MisStage][warn] stage_ids length mismatch: stage={stage_ids.shape[0]} labels={y.shape[0]}")
            return

        fp_idx = np.where((y == 0) & (preds == 1))[0]
        fn_idx = np.where((y == 1) & (preds == 0))[0]

        def _name(sid: int) -> str:
            if sid < 0:
                return 'UNKNOWN'
            if isinstance(stage_id2name, (list, tuple)) and sid < len(stage_id2name):
                return str(stage_id2name[sid])
            return f'stage_id={sid}'

        def _summarize(idxs: np.ndarray, tag: str):
            if idxs.size == 0:
                self._emit_debug(f"[MisStage] {tag}: 0")
                return

            names = np.array([_name(int(stage_ids[i])) for i in idxs], dtype=object)
            uniq, cnt = np.unique(names, return_counts=True)
            order = np.argsort(-cnt)
            top = [(str(uniq[i]), int(cnt[i])) for i in order[:10]]
            self._emit_debug(f"[MisStage] {tag}: {int(idxs.size)} | top={top}")

            n = int(min(int(max_print), int(idxs.size)))
            for j in idxs[:n]:
                self._emit_debug(
                    f"  [{tag}] idx={int(j)} stage={_name(int(stage_ids[j]))} y={int(y[j])} pred={int(preds[j])} prob={float(p[j]):.6f}"
                )

        _summarize(fp_idx, 'FP')
        _summarize(fn_idx, 'FN')

        # Print mapping once for traceability.
        if isinstance(stage_id2name, (list, tuple)) and len(stage_id2name) > 0:
            try:
                preview = [(int(i), str(stage_id2name[i])) for i in range(min(10, len(stage_id2name)))]
                self._emit_debug(f"[MisStage] stage_id2name (preview)={preview}")
            except Exception:
                pass
    
    @torch.no_grad()
    def get_probs(self, logits_dict={}):
        probs_dict = {}
        for o_r in logits_dict:
            probs_dict[o_r] = logits_dict[o_r].softmax(1)[:, 1]
        return probs_dict

    @torch.no_grad()
    def _single_eval(self, labels, probs):
        score = {}
        with torch.no_grad():
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            if torch.is_tensor(probs):
                probs = probs.cpu().numpy()
            # If a split has only one class (e.g., all-benign), sklearn's AUROC/AUPRC will raise.
            # For our flowing-day splits this is common; return NaN for AUROC/AUPRC in that case.
            try:
                uniq = np.unique(labels)
                single_class = (uniq.size <= 1)
            except Exception:
                single_class = False
            if getattr(self.args, 'print_cm', False):
                # quick distribution sanity to debug "best_th stuck" problems
                try:
                    pos_rate = float(labels.mean())
                except Exception:
                    pos_rate = None
                msg = (
                    f"[EvalDebug] n={len(labels)} pos_rate={pos_rate} "
                    f"prob_min={float(np.min(probs)):.6f} prob_p50={float(np.median(probs)):.6f} "
                    f"prob_p90={float(np.quantile(probs, 0.9)):.6f} prob_max={float(np.max(probs)):.6f}"
                )
                self._emit_debug(msg)

                best_f1, best_th, cm_best = get_best_f1_and_cm(labels, probs)
                # NOTE: MacroF1 stores POSITIVE-class F1
                score['MacroF1'] = best_f1
                score['BestThres'] = best_th
                score['CM'] = cm_best

                # always print a fixed set of thresholds around the observed probability range
                fixed_parts = []
                for th in get_fixed_threshold_points():
                    p, r, f1p, cm_th = _prf_at_threshold(labels, probs, float(th))
                    tn, fp, fn, tp = cm_th.ravel()
                    fixed_parts.append(
                        f"th={th:.2f} P={p:.4f} R={r:.4f} F1={f1p:.4f} cm=[[{tn},{fp}],[{fn},{tp}]]"
                    )
                msg = "[ThreshFixed] " + " | ".join(fixed_parts)
                self._emit_debug(msg)

                # show top-k thresholds (by positive-class F1)
                topk = get_topk_thresholds(labels, probs, k=5)
                pretty = []
                for row in topk:
                    tn, fp, fn, tp = row['cm'].ravel()
                    pretty.append(
                        f"th={row['th']:.2f} P={row['p_pos']:.4f} R={row['r_pos']:.4f} F1={row['f1_pos']:.4f} "
                        f"cm=[[{tn},{fp}],[{fn},{tp}]]"
                    )
                msg = "[ThreshTopK_posF1] " + " | ".join(pretty)
                self._emit_debug(msg)

                # fixed-FPR points are often more meaningful for anomaly detection
                for fpr_tgt in (0.01, 0.05):
                    try:
                        p_f, r_f, f1_f, th_f, cm_f = _prf_at_fpr(labels, probs, target_fpr=fpr_tgt)
                        tn, fp, fn, tp = cm_f.ravel()
                        msg = (
                            f"[FixedFPR] target={fpr_tgt:.2%} th={th_f:.6f} "
                            f"P={p_f:.4f} R={r_f:.4f} F1={f1_f:.4f} cm=[[{tn},{fp}],[{fn},{tp}]]"
                        )
                        self._emit_debug(msg)
                    except Exception:
                        # don't let debug reporting crash evaluation
                        pass
            else:
                score['MacroF1'] = get_best_f1(labels, probs)[0]

            if single_class:
                score['AUROC'] = float('nan')
                score['AUPRC'] = float('nan')
            else:
                score['AUROC'] = roc_auc_score(labels, probs)
                score['AUPRC'] = average_precision_score(labels, probs)

            if self._debug_log_fp is not None:
                self._debug_log_fp.flush()

        return score
    
    @torch.no_grad()
    def eval(self, labels_dict, probs_dict):
        result = {}
        for k in self.output_route:
            result[k] = self._single_eval(labels_dict[k], probs_dict[k])
        return result

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr_ft, weight_decay=self.args.l2_ft)
        score_test = None
        best_state_dict = None
        best_epoch = -1
        best_score_test = None
        for epoch in tqdm( range(self.args.epoch_ft) ):
            loss_items_total_train = {k:0 for k in self.output_route }
            total_loss_graph = 0
            total_loss_node = 0
            for batched_data in self.train_dataloader:
                batched_graph, batched_labels_dict, batched_khop_graph = batched_data
                # FIXME: device issue?
                batched_graph = batched_graph.to(self.args.device)
                for k,v in batched_labels_dict.items():
                    batched_labels_dict[k] = v.to(self.args.device)
                batched_khop_graph = batched_khop_graph.to(self.args.device)

                self.model.train()
                lambda_contrast = float(getattr(self.args, 'lambda_contrast', 0.0) or 0.0)
                if lambda_contrast > 0:
                    # Two stochastic views via dropout randomness (keep model in train mode)
                    logits_dict_1, emb_1 = self.model(
                        batched_graph,
                        batched_graph.ndata['feature'],
                        batched_khop_graph,
                        scen='train',
                        return_emb=True,
                    )
                    logits_dict_2, emb_2 = self.model(
                        batched_graph,
                        batched_graph.ndata['feature'],
                        batched_khop_graph,
                        scen='train',
                        return_emb=True,
                    )
                    loss, loss_items = self.get_loss(logits_dict_1, labels_dict=batched_labels_dict)
                    route = str(getattr(self.args, 'contrast_on', 'e'))
                    z1 = emb_1.get(route, None) if isinstance(emb_1, dict) else None
                    z2 = emb_2.get(route, None) if isinstance(emb_2, dict) else None
                    if z1 is not None and z2 is not None:
                        # avoid O(N^2) sim-matrix memory blow-up
                        z1, z2 = _subsample_for_contrast(
                            z1,
                            z2,
                            max_samples=int(getattr(self.args, 'max_contrast_samples', 2048) or 2048),
                            seed=int(getattr(self.args, 'seed', 0) or 0) + int(epoch),
                        )
                        contrast_loss = _nt_xent_loss(
                            z1,
                            z2,
                            tau=float(getattr(self.args, 'contrast_tau', 0.2)),
                            eps=1e-12,
                        )
                        loss = loss + lambda_contrast * contrast_loss
                        # add to loss_items for logging
                        loss_items = dict(loss_items)
                        loss_items['contrast'] = float(contrast_loss.detach().cpu().item())
                    else:
                        loss_items = dict(loss_items)
                        loss_items['contrast'] = 0.0
                    # cleanup
                    del logits_dict_2, emb_2
                    del emb_1
                    logits_dict = logits_dict_1
                else:
                    logits_dict = self.model(batched_graph, batched_graph.ndata['feature'], batched_khop_graph, scen='train')
                    loss, loss_items = self.get_loss(logits_dict, labels_dict=batched_labels_dict)

                for k in loss_items_total_train:
                    loss_items_total_train[k[0]] += loss_items[k]

                optimizer.zero_grad()
                loss.backward()
                # pcgrad_fn(self.model, losses=loss, optimizer=optimizer)

                optimizer.step()
                # scheduler.step()
                # # The following code is used to record the memory usage
                # py_process = psutil.Process(os.getpid())
                # print(f"CPU Memory Usage: {py_process.memory_info().rss / (1024 ** 3)} GB")
                # print(f"GPU Memory Usage: {torch.cuda.memory_reserved() / (1024 ** 3)} GB")

                # clear GPU cache
                del batched_data
                del batched_graph
                del batched_labels_dict
                del batched_khop_graph
                del logits_dict
                del loss
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                labels_dict_val_mul = {k:[] for k in self.output_route }
                probs_dict_val_mul = {k:[] for k in self.output_route }
                loss_items_total_val = {k:0 for k in self.output_route }
                # eval loop
                for batched_data in self.val_dataloader:
                    batched_graph, batched_labels_dict, batched_khop_graph = batched_data
                    # FIXME: device issue?
                    batched_graph = batched_graph.to(self.args.device)
                    for k,v in batched_labels_dict.items():
                        batched_labels_dict[k] = v.to(self.args.device)
                        if k[0] in self.output_route:
                            labels_dict_val_mul[k[0]].append(v)
                    batched_khop_graph = batched_khop_graph.to(self.args.device)
                    self.model.eval()
                    with torch.no_grad():
                        logits_dict = self.model(batched_graph, batched_graph.ndata['feature'], batched_khop_graph, scen='val')
                        _, loss_items = self.get_loss(logits_dict, labels_dict=batched_labels_dict)
                        for k in self.output_route:
                            loss_items_total_val[k] += loss_items[k]

                        probs = self.get_probs(logits_dict)
                        for k in probs:
                            probs_dict_val_mul[k].append(probs[k])
                    
                    del batched_data
                    del batched_graph
                    del batched_labels_dict
                    del batched_khop_graph
                    del logits_dict
                    del probs
                with torch.no_grad():
                    for k in self.output_route:
                        labels_dict_val_mul[k] = torch.cat([t for t in labels_dict_val_mul[k]])
                        probs_dict_val_mul[k] = torch.cat([t for t in probs_dict_val_mul[k]])
                    # get eval score
                    score_val = self.eval(labels_dict_val_mul, probs_dict_val_mul)

                    del labels_dict_val_mul
                    del probs_dict_val_mul
                    # average different scores
                    score_overall_val = 0
                    for k in self.output_route:
                        score_overall_val += score_val[k][self.args.metric]
                    score_overall_val /= len(self.output_route)

                    if getattr(self.args, 'print_cm', False):
                        for k in self.output_route:
                            if 'CM' in score_val[k]:
                                print(f"[Val CM] {NAME_MAP[k]} best_th={score_val[k].get('BestThres'):.2f} cm={score_val[k]['CM'].tolist()}")
                log_loss(['Train', 'Val'], [loss_items_total_train, loss_items_total_val])
                del loss_items_total_train
                del loss_items_total_val

                # periodic one-line summary for trend tracking
                self._emit_epoch_summary(epoch=epoch, split='val', score_dict=score_val)

                # select the best on val set
                improved = (score_overall_val > self.best_score)
                if improved:
                    self.best_score = score_overall_val
                    self.patience_knt = 0
                    best_epoch = int(epoch)
                    # keep best weights
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

                    # run test snapshot for visibility (and keep it as current best)
                    labels_dict_test_mul, probs_dict_test_mul, _ = self._run_split_eval(self.test_dataloader, scen='test')
                    score_test = self.eval(labels_dict_test_mul, probs_dict_test_mul)
                    best_score_test = score_test
                    del labels_dict_test_mul
                    del probs_dict_test_mul

                    # log to stdin
                    print(f'Epoch {epoch}: best_val={self.best_score} (update)\n{pprint.pformat(score_test)}')
                else:
                    self.patience_knt += 1
                    if self.patience_knt > self.args.patience:
                        print(f"[EarlyStop] patience exceeded at epoch={epoch}, best_epoch={best_epoch}, best_val={self.best_score}")
                        break

        # Final test: load the best epoch weights (if any), then evaluate once.
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        labels_dict_test_mul, probs_dict_test_mul, _ = self._run_split_eval(self.test_dataloader, scen='test')
        score_test = self.eval(labels_dict_test_mul, probs_dict_test_mul)

        del labels_dict_test_mul
        del probs_dict_test_mul

        if getattr(self.args, 'print_cm', False):
            print(f"[BestModel] best_epoch={best_epoch} best_val={self.best_score}")

        # Print + write final thresholded metrics (P/R/F1/CM) to debug log.
        self._final_test_report(score_test)

        if self._debug_log_fp is not None:
            try:
                self._debug_log_fp.flush()
                self._debug_log_fp.close()
            except Exception:
                pass
            self._debug_log_fp = None

        return score_test