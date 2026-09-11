# Pretrained artifacts

These files are the inference artifacts already present in the project:

- `gru_model.h5`: Keras GRU model.
- `feature_scaler.pkl`: fitted MinMaxScaler for `VDMR`, `VTPK`, `VZMX`, `VTZA`.
- `target_scaler.pkl`: fitted MinMaxScaler for `VHM0`.

They are included so users can run inference without retraining. The model was developed with TensorFlow/Keras and should be loaded in a compatible runtime.
