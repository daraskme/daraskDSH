"""Single-image temporal reference conditioning for Anima Edit checkpoints."""
import torch
from comfy.model_base import Anima
from comfy.patcher_extension import WrappersMP

class AnimaEditReference:
    CATEGORY = "conditioning/Anima"
    FUNCTION = "apply"
    RETURN_TYPES = ("MODEL",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "latent": ("LATENT",)}}

    def apply(self, model, latent):
        if not isinstance(model.model, Anima):
            raise ValueError("Anima Edit requires an Anima base model.")
        reference = model.get_model_object("process_latent_in")(latent["samples"])
        if reference.ndim == 4:
            reference = reference.unsqueeze(2)
        if reference.shape[:1] != (1,) or reference.shape[2] != 1:
            raise ValueError("Choose one reference image for Anima Edit.")

        def sample_with_reference(executor, x, *args, **kwargs):
            ref = reference.to(x).expand(x.shape[0], -1, -1, -1, -1)
            if x.shape[-2:] != ref.shape[-2:]:
                raise ValueError("The sampled latent and reference must have the same size.")
            frames = torch.cat((x, ref), dim=2)
            return executor(frames, *args, **kwargs).narrow(2, 0, x.shape[2])

        patched = model.clone()
        patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "anima_edit_single_reference", sample_with_reference)
        return (patched,)

NODE_CLASS_MAPPINGS = {"AnimaEditSingleReference": AnimaEditReference}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaEditSingleReference": "Anima Edit · 参照画像"}
