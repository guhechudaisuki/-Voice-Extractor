"""Single-pass, shrinkage-regularized episode whitening; never serialized."""
import numpy as np


class EpisodeWhitening:
    def __init__(self, vectors, groups=None):
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError('Whitening requires finite episode observations')
        self.count = len(matrix)
        self.fit_scope = 'impostor_within_cluster' if groups is not None else 'unlabeled_covariance'
        self.group_count = len(set(groups)) if groups is not None else 0
        if groups is not None and len(groups) != len(matrix):
            raise ValueError('One cohort group is required for every whitening observation')
        self.applied = False
        self.center = np.zeros(matrix.shape[1])
        self.basis = np.empty((0,matrix.shape[1]))
        self.gains = np.empty(0)
        if len(matrix) < 8:
            return
        matrix = matrix / np.maximum(np.linalg.norm(matrix,axis=1,keepdims=True),1e-12)
        if groups is not None:
            # Equal cohort mass, using within-speaker residuals only. Neither a
            # prolific target nor between-speaker identity offsets fit this map.
            cohorts = []
            for group in sorted(set(groups),key=str)[:8]:
                rows = matrix[np.asarray(groups) == group]
                if len(rows) < 2:
                    continue
                rows = rows[np.linspace(0,len(rows)-1,min(16,len(rows))).astype(int)]
                cohorts.append((rows-np.median(rows,axis=0))/np.sqrt(len(rows)-1))
            if len(cohorts) < 2:
                return
            residual = np.concatenate(cohorts)/np.sqrt(len(cohorts))
            denominator = 1.
        else:
            if len(matrix) > 128:
                matrix = matrix[np.linspace(0,len(matrix)-1,128).astype(int)]
            residual = matrix-np.median(matrix,axis=0)
            denominator = max(1,len(matrix)-1)
        _, singular, basis = np.linalg.svd(residual,full_matrices=False)
        variance = singular**2/denominator
        if float(variance.sum()) <= 1e-12:
            return
        prior = max(float(variance.sum()/matrix.shape[1]),1e-4)
        # Shrink towards isotropy and cap condition number. Common channel
        # directions are attenuated without amplifying unseen dimensions.
        eigenvalues = .5*variance+.5*prior
        self.gains = np.clip(np.sqrt(prior/eigenvalues),.5,1.)-1.
        self.basis = basis
        # Estimate covariance about the sample location, but never translate
        # queries/references by an unlabeled speaker-dependent population mean.

    def transform(self, vectors):
        self.applied = bool(len(self.basis))
        x = np.asarray(vectors,dtype=np.float64)
        x = x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
        if len(self.basis):
            x = x + ((x @ self.basis.T)*self.gains) @ self.basis
        return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

    def clear(self):
        self.applied = False
        self.center.fill(0)
        self.basis = np.empty((0,len(self.center)))
        self.gains = np.empty(0)

    def diagnostics(self):
        return {'method':'single_pass_shrinkage_whitening','observations':self.count,
                'fit_scope':self.fit_scope,'cohort_groups':self.group_count,
                'fitted':bool(len(self.basis)), 'applied':self.applied,
                'active':self.applied, 'diagnostic_only':not self.applied, 'state_persisted':False}
