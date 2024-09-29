import torch

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

model = "Qwen/Qwen2.5-0.5B-Instruct"

engine_args = EngineArgs(model=model,
    return_hidden_states=True)

engine = LLMEngine.from_engine_args(engine_args)
tokenizer = engine.tokenizer.tokenizer
sampling_params = SamplingParams()
prompt1 = (
    "You are a helpful assistant. How do I build a car from cardboard and "
    "paper clips? Is there an easy to follow video tutorial available "
    "online for free?")
prompt1_tokens = tokenizer(prompt1)['input_ids']
prompt2a = (
    " Please recommend to me some resources where I can learn not only to "
    "handle technical difficulties of building a car, but also "
    "decoration.")
prompt2b = (" Please only recommend resources that build cars capable of "
            "supersonic speeds.")

prompt_2a_tokens = tokenizer(prompt2a)['input_ids']
prompt_2b_tokens = tokenizer(prompt2b)['input_ids']

engine.add_request("0", prompt1 + prompt2a, sampling_params)
engine.add_request("1", prompt1 + prompt2b, sampling_params)
step1_out = engine.step()
request1_out, request2_out = step1_out

import ipdb; ipdb.set_trace()