# EMG Gesture Recognition

Experiments on the UCI EMG Gestures dataset (36 subjects, 6 gestures, 8 channels at 200Hz). Random chance = 16.7%.

---

## pyTorch Conv1d Autoencoder

**Approach:** Trains one Conv1d autoencoder per gesture class (6 total). Anomaly detection is performed by comparing normalized reconstruction errors across all 6 models; the gesture whose autoencoder produced the lowest error is the predicted class.

**Best result:** 34% accuracy. Gestures 4 and 5 reached ~62% individually. Gestures 1, 2, and 3 were consistently confused with one another and with whichever model had the lowest baseline error.

**Issues:** The autoencoder learned a generalized representation of forearm muscle activity rather than gesture-specific patterns. Because error differences between gestures were small (~10–20%), normalization failed to reliably separate them.

**Attempted solutions:**
- PCA of bottleneck representations required 50 components to explain 80% of variance, indicating gesture information was spread across a very high-dimensional space with no clear cluster separation
- ICA preprocessing introduced inconsistent scaling across models and worsened accuracy in every configuration tested
- Reducing bottleneck compression had adverse effects on results

---

## pyTorch Conv1d Classifier

**Approach:** Single Conv1d classifier trained on all 6 gesture classes simultaneously using cross-entropy loss. Architecture consists of a 3-layer Conv1d encoder (8→16→32→16 channels, 16x time compression to bottleneck of 800 values) followed by a fully connected classification head (800→128→6). Subject-independent train/val/test split: subjects 1–30 series 1–3 for training (540 samples), series 4 for validation (180 samples), subjects 31–36 for final evaluation (144 samples).

**Best result:** 27.4% accuracy.

**Issues:** Severe overfitting due to a small dataset. Val loss increased monotonically after ~60 epochs regardless of learning rate. Early stopping with patience=50 and dropout (0.2 encoder, 0.3 head) provided marginal improvement but did not resolve the underlying data limitation.

**Attempted solutions:** Dropout regularization (0.2–0.5), early stopping, learning rate reduction (1e-3 to 1e-4), and reduced classification head capacity. All produced marginal and inconsistent improvements. The persistent confusion between gestures 1, 2, and 3 across both the autoencoder and classifier approaches suggests these gestures are not well separated in the time-domain signal space.

---

## Sklearn Random Forest

**Approach:** Single Random Forest classifier. Two feature representations were evaluated: (1) flattened raw signal windows and (2) handcrafted time-domain features.

**Best result:** 91.0% test accuracy with handcrafted features. Gesture 1 (rest) achieved perfect classification at 100% F1. Gestures 2, 4, and 5 reached 92–94% F1. Gestures 3 and 6 were the most challenging at 82–84% F1.

**Raw signal result:** 80.6% test accuracy with flattened 3200-feature representation. Counterintuitively worse than handcrafted features despite containing more information, likely that the Random Forest had insufficient training samples.

**Issues:** Remaining confusion concentrated in gesture pairs (3 vs. 6 and 4 vs. 5).
