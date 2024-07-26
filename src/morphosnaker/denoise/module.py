from .factory import create_config, create_model


class DenoiseModule:
    def __init__(self, method="n2v", **config_kwargs):
        self.method = method
        self.config = create_config(method, **config_kwargs)
        self.model = create_model(method, self.config)

    def configurate(self, **config_kwargs):
        new_method = config_kwargs.pop("method", self.method)

        if new_method != self.method:
            # Create a new config with default values for the new method
            new_config = create_config(new_method)
            # Update the new config with any provided kwargs
            for key, value in config_kwargs.items():
                if hasattr(new_config, key):
                    setattr(new_config, key, value)
                else:
                    raise ValueError(
                        f"Invalid config parameter for method"
                        f"{new_method}: {key}"
                    )
        else:
            # Update existing config
            new_config = self._update_config(config_kwargs)

        if new_config != self.config:
            self.config = new_config
            self.method = new_method
            self.model = create_model(self.method, self.config)

        return self.config

    def _update_config(self, new_kwargs):
        updated_dict = self.config.__dict__.copy()
        for key, value in new_kwargs.items():
            if hasattr(self.config, key):
                updated_dict[key] = value
            else:
                raise ValueError(
                    f"Invalid config parameter for method {self.method}: {key}"
                )
        return create_config(self.method, **updated_dict)

    def train_2D(self, images):
        return self.model.train_2D(images)

    def train_3D(self, images):
        return self.model.train_3D(images)

    def predict(self, image):
        return self.model.predict(image)

    def load_model(self, path):
        self.model.load(path)

    def get_config(self):
        return self.config
