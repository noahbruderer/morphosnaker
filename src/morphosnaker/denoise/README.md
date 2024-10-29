# MorphoSnaker Denoising Module

The denoising module in MorphoSnaker provides advanced techniques for noise reduction in microscopy images.

## Features

- Noise2Void implementation
- Flexible configuration system
- Support for 2D and 3D images

## Usage

To use the denoising module, follow these steps:

1. Configure the denoising module:

```python
from morphosnaker import denoise

denoiser = denoise.config('n2v', denoising_mode='2D',
                n2v_patch_shape=(64, 64),
                train_steps_per_epoch=100,
                train_epochs=10,
                train_batch_size=128)
```

2. Train the model:

```python
training_history = denoiser.train_2D([image])
```

3. Denoise an image:

```python
denoised_image = denoiser.denoise(noisy_image)
```

4. Save the trained model:

```python
denoiser.save_model('/path/to/save/model')
```

5. Load a trained model:

```python
denoiser.load_model('/path/to/saved/model')
```

For a full list of configuration options, refer to the Noise2VoidTrainingConfig and Noise2VoidPredictionConfig classes.

## Extending the Module

To add new denoising methods:

1. Create a new subclass of DenoiseMethodModuleBase.
2. Implement the required methods.
3. Update the factory functions in factory.py.
4. Add the new method to the DenoiseModule.config() method.