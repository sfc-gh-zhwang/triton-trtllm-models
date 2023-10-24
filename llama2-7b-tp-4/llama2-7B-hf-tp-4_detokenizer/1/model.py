from pathlib import Path
from typing import List

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


class TritonPythonModel:
    """
    Python backend logic to load and run tokenizer of queries.
    The class name has to be TritonPythonModel.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        tokenizer_dir = Path(__file__).parent.resolve()
        logger = pb_utils.Logger
        logger.log(f'Initializing tokenizer from {tokenizer_dir}')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        logger.log(f'Initialized tokenizer from {tokenizer_dir}')

    # I wanted to type annotation as List[pb_utils.InferenceRequest] but it seems to
    # cause issues, see https://github.com/triton-inference-server/server/issues/4743.
    def execute(self, requests: List) -> List:
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.
        for request in requests:
            # Perform inference on the request and append it to responses list...
            tokens_batch = pb_utils.get_input_tensor_by_name(request, "tokens_batch")
            tokens_batch = tokens_batch.as_numpy()
            input_sequence_lengths = pb_utils.get_input_tensor_by_name(request, "input_sequence_lengths")
            input_sequence_lengths = input_sequence_lengths.as_numpy()
            output = []
            for input_sequence_length, batch in zip(input_sequence_lengths, tokens_batch):
                output.append(
                    [self.tokenizer.decode(token_ids[input_sequence_length:], skip_special_tokens=True)
                     for token_ids in batch])
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("output", np.array(output).astype(object))
            ]))

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses
