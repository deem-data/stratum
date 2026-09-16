"""Fit-pass planning: what the fitting pass is allowed to skip, and what it is not.

The dangerous half of this is the second one. Marking a step dead because nothing reads
its output in the fitting pass would leave a step that has to be *fitted* unfitted, and
the plan would then score a model built from an unfitted transformer without saying so.
"""
import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold

import stratum as st
from stratum.optimizer._optimize import SearchConfig, optimize
from stratum.optimizer.logical._candidate_ops import ScoreCandidatesOp
from stratum.optimizer.logical._ops import BaseEstimatorOp
from stratum.optimizer.logical._scoring import resolve_scoring
from stratum.runtime._scheduler import SequentialScheduler
from stratum.frontend._skrub_graph import get_data

N_SPLITS = 3
SCORING = "neg_mean_squared_error"

TAIL_CALLS: list = []


def _spy_tail(values):
    """A post-processing step that records every pass it is run in."""
    TAIL_CALLS.append(np.shape(values))
    return values


class ShiftTail(TransformerMixin, BaseEstimator):
    """A tail whose fitted state changes the score, so skipping its fit would show."""

    def fit(self, X, y=None):
        self.shift_ = float(np.mean(np.asarray(X)))
        return self

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X).transform(X)

    def transform(self, X):
        return np.asarray(X) - self.shift_


class RecordingTail(TransformerMixin, BaseEstimator):
    """A step after the predictor that has to be fitted on the predictor's output."""

    FITS: list = []

    def fit(self, X, y=None):
        RecordingTail.FITS.append("fit")
        return self

    def fit_transform(self, X, y=None, **kwargs):
        RecordingTail.FITS.append("fit_transform")
        return X

    def transform(self, X):
        return X


class FitPassPlanningTest(unittest.TestCase):
    def setUp(self):
        # An eager preview would run the plan at construction time and count as a pass.
        self.enterContext(st.config_context(eager_data_ops=False))
        rng = np.random.default_rng(0)
        self.df = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
        self.df["t"] = 2.0 * self.df["a"] - self.df["b"]
        TAIL_CALLS.clear()
        RecordingTail.FITS.clear()

    def _pipeline(self, tail=None):
        data = st.as_data_op(self.df)
        y = data["t"].skb.mark_as_y()
        X = data[["a", "b"]].skb.mark_as_X(cv=KFold(n_splits=N_SPLITS), split_kwargs={})
        pred = X.skb.apply(DummyRegressor(), y=y)
        return pred if tail is None else tail(pred)

    def _plan(self, tail=None):
        return self._compile(tail)[0]

    def _compile(self, tail=None):
        dag = self._pipeline(tail)
        return optimize(dag, env=get_data(dag),
                        search=SearchConfig(resolve_scoring(SCORING)))

    def _scores(self, tail=None, marked=True):
        """Run the plan with fit-pass skipping on or off."""
        linearized, split_pos, flagged = self._compile(tail)
        if not marked:
            for op in linearized:
                op.dead_in_fit = False
        sched = SequentialScheduler(linearized, split_pos, flagged)
        sched.grid_search(KFold(n_splits=N_SPLITS))
        return sched.results_["scores"].to_list()

    def test_post_processing_runs_once_per_fold_not_twice(self):
        """The tail's output is read only when the fold is scored, so the fitting pass
        does not run it. It used to run in both passes and be dropped unread in one."""
        st._api.grid_search(self._pipeline(lambda p: p.skb.apply_func(_spy_tail)),
                            scoring=SCORING)
        self.assertEqual(len(TAIL_CALLS), N_SPLITS)

    def test_the_candidate_set_is_dead_while_fitting(self):
        self.assertTrue(self._plan()[-1].dead_in_fit)

    def test_a_step_that_must_be_fitted_is_never_marked(self):
        """A transformer after the predictor is fitted on the predictor's output, so it
        has to run in the fitting pass even though nothing there reads its result."""
        plan = self._plan(lambda p: p.skb.apply(RecordingTail(), how="no_wrap"))
        tails = [op for op in plan if isinstance(op, BaseEstimatorOp)]
        self.assertTrue(tails, "expected the estimator ops in the plan")
        for op in tails:
            self.assertFalse(op.dead_in_fit, f"{op} would go unfitted")

        st._api.grid_search(self._pipeline(
            lambda p: p.skb.apply(RecordingTail(), how="no_wrap")), scoring=SCORING)
        self.assertEqual(len(RecordingTail.FITS), N_SPLITS)

    def test_work_feeding_the_estimator_is_never_marked(self):
        """Everything upstream of a fitted step is needed to fit it."""
        plan = self._plan()
        split_pos = next(i for i, op in enumerate(plan) if op.is_split_op)
        upstream = [op for op in plan[split_pos:]
                    if not isinstance(op, ScoreCandidatesOp)
                    and any(isinstance(o, BaseEstimatorOp) for o in op.outputs)]
        self.assertTrue(upstream, "expected ops feeding the estimator")
        for op in upstream:
            self.assertFalse(op.dead_in_fit)

    def test_scores_are_unchanged_by_the_skipping(self):
        """Equivalence: skipping work the fitting pass does not need must not move a
        number. `ShiftTail` carries fitted state into the score, so a step wrongly
        skipped would either raise or shift the result rather than pass quietly."""
        tail = lambda p: p.skb.apply(ShiftTail(), how="no_wrap")
        np.testing.assert_allclose(self._scores(tail, marked=True),
                                   self._scores(tail, marked=False), rtol=1e-12)

    def test_plain_plan_scores_are_unchanged_by_the_skipping(self):
        np.testing.assert_allclose(self._scores(marked=True),
                                   self._scores(marked=False), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
