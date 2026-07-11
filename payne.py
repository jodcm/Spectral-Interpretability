"""
payne.py
========
EN: A compact "The Payne"-style emulator (Ting et al. 2019) as the project's SECOND
    emulator, next to TransformerPayne. The Payne is a fully-connected neural network
    that maps stellar labels (Teff, log g, abundances) directly to a spectrum. The
    classic recipe trains it on a library of synthetic spectra from a physical code;
    here we use TransformerPayne as that spectral library (it plays the role of the
    ground-truth physics), so this Payne is a fast MLP surrogate of TransformerPayne on
    our 4000-5000 A grid. That lets us compare the two emulators fairly: does an RF
    trained on Payne-generated spectra transfer to real DESI as well as one trained on
    TransformerPayne spectra?
ES: Un emulador compacto estilo "The Payne" (Ting et al. 2019) como SEGUNDO emulador
    del proyecto, junto a TransformerPayne. The Payne es una red neuronal totalmente
    conectada que mapea etiquetas estelares (Teff, log g, abundancias) directamente a
    un espectro. La receta clasica lo entrena con una libreria de espectros sinteticos
    de un codigo fisico; aca usamos TransformerPayne como esa libreria (hace de fisica
    verdadera), asi este Payne es un sustituto MLP rapido de TransformerPayne en la
    grilla 4000-5000 A. Permite comparar los dos emuladores de forma justa.

Requires / Requiere: scikit-learn (MLPRegressor). No deep-learning framework needed.
"""
import numpy as np
import project_lib as P

# EN: label vector order (must be stable) | ES: orden del vector de etiquetas (estable)
LABEL_KEYS = ["logteff", "logg"] + list(P.VARIED_ELEMENTS)


def labels_to_matrix(dicts):
    """EN: list of label dicts -> (n, n_labels) matrix in LABEL_KEYS order.
    ES: lista de dicts de etiquetas -> matriz (n, n_labels) en orden LABEL_KEYS."""
    return np.array([[d[k] for k in LABEL_KEYS] for d in dicts], dtype=float)


class PayneEmulator:
    """EN: MLP emulator labels -> normalized spectrum on P.WAVE_GRID.
    ES: emulador MLP etiquetas -> espectro normalizado en P.WAVE_GRID."""

    def __init__(self, hidden_layer_sizes=(300, 300), max_iter=300, random_state=0):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.net = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                activation="tanh", solver="adam",
                                max_iter=max_iter, random_state=random_state,
                                early_stopping=True, n_iter_no_change=15)
        self.wave_grid = P.WAVE_GRID

    def fit(self, dicts, spectra):
        """EN: dicts = labels, spectra = (n, len(wave_grid)) normalized intensities.
        ES: dicts = etiquetas, spectra = (n, len(wave_grid)) intensidades normalizadas."""
        Xlab = self.scaler.fit_transform(labels_to_matrix(dicts))
        self.net.fit(Xlab, np.asarray(spectra, dtype=float))
        return self

    def predict_spectra(self, dicts):
        """EN: labels -> predicted normalized spectra (n, len(wave_grid)).
        ES: etiquetas -> espectros normalizados predichos (n, len(wave_grid))."""
        Xlab = self.scaler.transform(labels_to_matrix(dicts))
        return self.net.predict(Xlab)

    def save(self, path):
        import joblib
        joblib.dump({"scaler": self.scaler, "net": self.net,
                     "wave_grid": self.wave_grid, "label_keys": LABEL_KEYS}, path)

    @staticmethod
    def load(path):
        import joblib
        b = joblib.load(path)
        obj = PayneEmulator.__new__(PayneEmulator)
        obj.scaler = b["scaler"]; obj.net = b["net"]; obj.wave_grid = b["wave_grid"]
        return obj
