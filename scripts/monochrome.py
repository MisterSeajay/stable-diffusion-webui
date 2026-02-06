"""
The Image I/O and Post-Processing system is responsible for handling image file
operations, metadata management, and image manipulation tasks in the Stable
Diffusion Web UI. This system enables loading and saving images with generation
parameters, creating image grids, manipulating image dimensions, and processing
image transparency.
"""
import gradio as gr
from PIL import Image
# SDWebUI imports
import modules.scripts as scripts
from modules import processing, shared, images, devices
from modules.processing import process_images, Processed
from modules.shared import opts, state


class Script(scripts.Script):
    def title(self):
        return "MakeMonochrome"

    def show(self, is_img2img: bool): # pyright: ignore[reportIncompatibleMethodOverride]
        return True  # Use `return True` to show this script in both txt2img and img2img
        #return is_img2img  # Use `return is_img2img` to show this script ONLY in the img2img tab
        #return scripts.AlwaysVisible  # for always-on scripts

    def ui(self, is_img2img) -> list[gr.components.Component]: # pyright: ignore[reportIncompatibleMethodOverride]
        with gr.Row(elem_id=self.elem_id("my_row")):
            info = gr.HTML("<p style=\"margin-bottom:0.75em\">MakeMonochrome: Ensure that black & white images are monochrome.</p>")
        with gr.Row(elem_id=self.elem_id("my_row")):
            mono = gr.Checkbox(False, label="Always force monochrome")
            keep = gr.Checkbox(False, label="Keep original generations")
        return [info, mono, keep]

    def run(self, p, info, mono, keep): # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Use Pillow to force the image being processed to be black and white. It
        relies on detecting certain words, e.g. "B&W" in the prompt. However if
        dynamic prompting is being used, these indicators might not be in the
        original prompt until after `process_images` is called. Thus, we check
        both the prompt AND the style name to try to catch this.

        Args:
            p: StableDiffusionProcessing object
            + All arguments exported by `ui` method
        """

        def desaturate(image):
            return image.convert("L")

        if "B&W" in p.prompt:
            mono = True

        if any("B&W" in s for s in p.styles):
            mono = True

        if mono and not keep:
            p.do_not_save_samples = True

        proc = process_images(p)

        for i in range(len(proc.images)):
            mono_this = False
            if not mono and "B&W" in proc.prompt:
                print("MakeMonochrome: B&W detected in prompt AFTER processing.")
                mono_this = True
            if mono or mono_this:
                proc.images[i] = desaturate(proc.images[i])
                images.save_image(
                    proc.images[i],
                    p.outpath_samples,
                    "",  # No prefix on the file name
                    proc.seed + i,
                    proc.prompt,
                    opts.samples_format,
                    info = proc.info,
                    p = p
                )

        return proc

