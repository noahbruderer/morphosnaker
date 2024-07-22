# import os
# import json
# import uuid
# import logging

# from ..experiment_manager.abc import ExperimentManagerBase
# from termcolor import colored
# import tifffile as tiff

# class SegmentationExperimentManager(ExperimentManagerBase):
#     def save_experiment_setup(self, segmentation_instance, method, image_path, **parameters):
#         image_name = os.path.basename(image_path)
#         self.metadata['input'].append({
#             'name': image_name,
#             'path': image_path,
#             'method': method,
#             'parameters': parameters
#         })
#         self.save_metadata()
        
#         image = segmentation_instance.load_image(image_path)
        
#         if method == 'cellpose':
#             self.log(f"Running Cellpose segmentation on {image_name}...", logging.INFO)
#             result = segmentation_instance.cellpose(image, **parameters)
#         elif method == 'stardist':
#             self.log(f"Running StarDist segmentation on {image_name}...", logging.INFO)
#             result = segmentation_instance.stardist(image, **parameters)
#         else:
#             raise ValueError(f"Unknown segmentation method: {method}")

#         result_path = os.path.join(self.output_dir, f"{os.path.splitext(image_name)[0]}_segmented.tiff")
#         tiff.imwrite(result_path, result)
        
#         self.update_metadata('results', {
#             'name': image_name,
#             'method': method,
#             'parameters': parameters,
#             'result_path': result_path
#         })

#         self.log(f"Segmentation completed for {image_name}. Results saved to {result_path}", logging.INFO)
#         return result