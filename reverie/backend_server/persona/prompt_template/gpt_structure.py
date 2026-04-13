"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import json
import random
import time
import requests

from utils import *

# NOTE: switched from OpenAI SDK to Ollama HTTP API.
# The original OpenAI usage is preserved but commented out for reference.
# import openai
# openai.api_key = openai_api_key

# Ollama server base URL (local). Update if your Ollama server runs elsewhere.
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_URL = OLLAMA_BASE_URL + "/api/generate"
# Models to use - switched to phi3:mini (2.2 GB) for memory efficiency
OLLAMA_CHAT_MODEL = "phi3:mini"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

def temp_sleep(seconds=0.1):
  time.sleep(seconds)


# def ollama_request(prompt, model=None, temperature=0, stream=False, timeout=600):
#   if model is None:
#     model = OLLAMA_CHAT_MODEL
  
#   import time
#   start_time = time.time()
#   print(f"[SENDING] OLLAMA: Sending request to model '{model}'...")
  
#   payload = {
#     "model": model,
#     "prompt": prompt,
#     "stream": stream,
#     # "format": "json",
#     "options": {
#       "temperature": temperature,
#       "num_gpu": 999,  # Force maximum GPU usage (RTX 3050 has 4GB VRAM)
#       "num_thread": 2   # Minimal CPU threads - let GPU handle it
#     }
#   }
def ollama_request(prompt, model=None, temperature=0, stream=False, timeout=60, max_tokens=None, stop=None):
  if model is None:
    model = OLLAMA_CHAT_MODEL
  
  import time
  start_time = time.time()
  print(f"[SENDING] OLLAMA: Sending request to model '{model}'...")
  
  options = {
    "temperature": temperature,
    "num_gpu": 999,
    "num_thread": 2
  }
  
  # Tell Ollama when to stop talking!
  if max_tokens:
    options["num_predict"] = max_tokens
  if stop:
    options["stop"] = stop

  payload = {
    "model": model,
    "prompt": prompt,
    "stream": stream,
    "options": options
  }
  try:
    resp = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    
    # Try to parse as JSON first
    try:
      data = resp.json()
      response = data.get("response") or data.get("text")
    except json.JSONDecodeError:
      # If JSON parsing fails, treat the entire response as plain text
      response = resp.text.strip()
    
    if not response:
      raise ValueError(f"Empty OLLAMA response")
    
    elapsed = time.time() - start_time
    print(f"[OK] OLLAMA: Response received in {elapsed:.1f}s\n")
    return response.strip() if response else ""  
  except requests.exceptions.ConnectionError as e:
    print(f"\n[ERROR] OLLAMA ERROR: Cannot connect to OLLAMA server")
    print(f"   Server URL: {OLLAMA_BASE_URL}")
    print(f"   REASON: Server is not running or not reachable")
    print(f"   FIX: Run 'ollama serve' in a separate terminal")
    print(f"   Details: {str(e)}\n")
    return "OLLAMA ERROR"
  except requests.exceptions.Timeout as e:
    print(f"\n[ERROR] OLLAMA ERROR: Request timeout after {timeout}s")
    print(f"   Model: {model}")
    print(f"   REASON: Server is slow or overloaded")
    print(f"   FIX: Check system resources or increase timeout")
    print(f"   Details: {str(e)}\n")
    return "OLLAMA ERROR"
  except requests.exceptions.HTTPError as e:
    status_code = e.response.status_code if hasattr(e, 'response') else 'Unknown'
    print(f"\n[ERROR] OLLAMA ERROR: HTTP {status_code}")
    print(f"   Model: {model}")
    print(f"   REASON: Model may not be installed or server responded with error")
    print(f"   FIX: Run 'ollama pull {model}' to install the model")
    print(f"   Details: {str(e)}\n")
    return "OLLAMA ERROR"
  except ValueError as e:
    print(f"\n[ERROR] OLLAMA ERROR: Invalid response format")
    print(f"   Model: {model}")
    print(f"   REASON: Server returned unexpected response structure")
    print(f"   Details: {str(e)}\n")
    return "OLLAMA ERROR"
  except Exception as e:
    print(f"\n[ERROR] OLLAMA ERROR: {type(e).__name__}")
    print(f"   Model: {model}")
    print(f"   REASON: {str(e)}")
    print(f"   TIP: Check OLLAMA server status and logs\n")
    return "OLLAMA ERROR"

def ChatGPT_single_request(prompt): 
  temp_sleep()
  # Use Ollama HTTP API instead of OpenAI SDK
  return ollama_request(prompt, model=OLLAMA_CHAT_MODEL, temperature=0)


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()

  try:
    # mapped to Ollama - using configured chat model
    return ollama_request(prompt, model=OLLAMA_CHAT_MODEL, temperature=0)
  except Exception as e:
    print(f"[ERROR] GPT4_request ERROR: {type(e).__name__}")
    print(f"   Message: {str(e)}")
    import traceback
    traceback.print_exc()
    return "ChatGPT ERROR"


def ChatGPT_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  try:
    return ollama_request(prompt, model=OLLAMA_CHAT_MODEL, temperature=0)
  except Exception as e:
    print(f"[ERROR] ChatGPT_request ERROR: {type(e).__name__}")
    print(f"   Message: {str(e)}")
    import traceback
    traceback.print_exc()
    return "ChatGPT ERROR"


def GPT4_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += json.dumps({"output": example_output})

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except Exception as e: 
      if verbose or i == repeat - 1:
        print(f"[GPT4_safe_generate_response Attempt {i+1}/{repeat}] {type(e).__name__}: {str(e)[:100]}")
      pass

  return False


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += json.dumps({"output": example_output})

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      
      # Try to extract JSON first, then fallback to plain text (phi3:mini returns plain text)
      try:
        end_index = curr_gpt_response.rfind('}') + 1
        if end_index > 0:
          curr_gpt_response = curr_gpt_response[:end_index]
          curr_gpt_response = json.loads(curr_gpt_response)["output"]
      except (json.JSONDecodeError, KeyError, ValueError):
        # If JSON parsing fails, use the response as plain text
        pass

      # print ("---ashdfaf")
      # print (curr_gpt_response)
      # print ("000asdfhia")
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except Exception as e: 
      if verbose or i == repeat - 1:
        print(f"[ChatGPT_safe_generate_response Attempt {i+1}/{repeat}] {type(e).__name__}: {str(e)[:100]}")
      pass

  return fail_safe_response


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose: 
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except Exception as e: 
      if verbose or i == repeat - 1:
        print(f"[ChatGPT_safe_generate_response_OLD Attempt {i+1}/{repeat}] {type(e).__name__}: {str(e)[:100]}")
      pass
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()
  try:
    # Map classic Completion API call to Ollama simple request. Use provided
    # temperature if available; engine/model fall back to configured chat model.
    model = OLLAMA_CHAT_MODEL
    temperature = gpt_parameter.get("temperature", 0)
    max_tokens = gpt_parameter.get("max_tokens", None)
    stop = gpt_parameter.get("stop", None)
    
    response = ollama_request(
        prompt, 
        model=model, 
        temperature=temperature, 
        max_tokens=max_tokens, 
        stop=stop
    )
    if "OLLAMA ERROR" in response:
      print(f"[ERROR] OLLAMA ERROR detected in GPT_request response")
      return "OLLAMA ERROR"
    return response
  except Exception as e:
    print (f"[ERROR] GPT_request EXCEPTION: {type(e).__name__}: {str(e)}")
    return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    try:
      curr_gpt_response = GPT_request(prompt, gpt_parameter)
      
      # Check for OLLAMA errors - don't try to validate OLLAMA errors
      if "OLLAMA ERROR" in curr_gpt_response or "TOKEN LIMIT" in curr_gpt_response:
        if verbose or i == repeat - 1:  # Show on last attempt or if verbose
          print(f"[Attempt {i+1}/{repeat}] Got OLLAMA/TOKEN error, retrying...")
        continue  # Try again
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose: 
        print (f"---- repeat count: {i}, validation failed")
        print (curr_gpt_response)
        print ("~~~~")
    except Exception as e:
      print(f"[Attempt {i+1}/{repeat}] Exception in safe_generate_response: {type(e).__name__}")
      if verbose:
        print(f"   Details: {str(e)}")
      continue
  
  print(f"[WARNING] safe_generate_response FAILED after {repeat} attempts")
  return fail_safe_response


def get_embedding(text, model="text-embedding-ada-002"):
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"

  OLLAMA_EMBED_URLS = [
    OLLAMA_BASE_URL + "/api/embeddings",
    OLLAMA_BASE_URL + "/api/embed",
  ]

  for url in OLLAMA_EMBED_URLS:
    try:
      # Use "prompt" key as expected by Ollama, with your embedding model
      resp = requests.post(url, json={"model": OLLAMA_EMBED_MODEL, "prompt": text}, timeout=60)
      resp.raise_for_status()
      data = resp.json()
      if isinstance(data, dict):
        if "embedding" in data:
          return data["embedding"]
        if "data" in data and isinstance(data["data"], list) and "embedding" in data["data"][0]:
          return data["data"][0]["embedding"]
      return data
    except Exception as e:
      last_exc = e

  print("[ERROR] OLLAMA EMBEDDING ERROR:", type(last_exc).__name__)
  print(f"   Message: {str(last_exc)}")
  # import traceback
  # traceback.print_exc()
  print(f"   Ollama took too long to respond. The system may be loading the model.")
  return "EMBEDDING ERROR"


if __name__ == '__main__':
  gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50, 
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0, 
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)





















# """
# Author: Joon Sung Park (joonspk@stanford.edu)
# Modified: Ollama backend with fast-fail timeouts, prompt truncation, and token limits.

# File: gpt_structure.py
# Description: Wrapper functions for calling Ollama (local LLM) APIs.
# """
# import json
# import random
# import time
# import requests

# from utils import *

# OLLAMA_BASE_URL = "http://localhost:11434"
# OLLAMA_GENERATE_URL = OLLAMA_BASE_URL + "/api/generate"
# OLLAMA_CHAT_MODEL = "phi3:mini"
# OLLAMA_EMBED_MODEL = "nomic-embed-text"

# # ── Timeout strategy ────────────────────────────────────────────────────────
# # phi3:mini on a 4 GB VRAM RTX 3050 takes ~5-15 s for short prompts and
# # ~30-60 s for long ones. The old 600 s default meant a single bad request
# # could hang the simulation for ~50 minutes (600 s × 5 retries).
# #
# # New values:
# #   SHORT  — simple single-line completions (wake_up_hour, pronunciatio …)
# #   MEDIUM — paragraph completions (hourly schedule, task decomp …)
# #   LONG   — multi-paragraph / conversation generation
# #   EMBED  — embedding requests (usually fast, but model load can be slow)
# #
# # If your machine is slower, raise these values; do NOT raise back to 600 s.
# TIMEOUT_SHORT  = 60    # seconds
# TIMEOUT_MEDIUM = 120
# TIMEOUT_LONG   = 180
# TIMEOUT_EMBED  = 60

# # Maximum prompt length (characters) sent to the model.
# # phi3:mini has a 4 k-token context window (~16 000 chars).
# # Keeping prompts under this limit avoids silent truncation / slowdowns.
# MAX_PROMPT_CHARS = 6000


# def temp_sleep(seconds=0.1):
#     time.sleep(seconds)


# def _truncate_prompt(prompt: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
#     """
#     Truncates a prompt that exceeds max_chars by keeping the first 60 % and
#     last 40 % of the allowed budget, with a visible marker in between.
#     This preserves both the instruction header and the final question which
#     the model needs to answer.
#     """
#     if len(prompt) <= max_chars:
#         return prompt
#     keep_head = int(max_chars * 0.60)
#     keep_tail = max_chars - keep_head
#     truncated = (
#         prompt[:keep_head]
#         + "\n... [TRUNCATED FOR LENGTH] ...\n"
#         + prompt[-keep_tail:]
#     )
#     print(f"[WARN] Prompt truncated: {len(prompt)} → {len(truncated)} chars")
#     return truncated


# def ollama_request(prompt, model=None, temperature=0, stream=False,
#                    timeout=TIMEOUT_MEDIUM, max_tokens=None, stop=None):
#     """
#     Send a single generation request to the local Ollama server.

#     Returns the model's response string, or "OLLAMA ERROR" on any failure.
#     Never raises — callers rely on the error sentinel string.
#     """
#     if model is None:
#         model = OLLAMA_CHAT_MODEL

#     prompt = _truncate_prompt(prompt)

#     start_time = time.time()
#     print(f"[SENDING] OLLAMA: model='{model}' prompt_len={len(prompt)} chars")

#     options = {
#         "temperature": temperature,
#         "num_gpu": 999,   # use all available VRAM
#         "num_thread": 2,  # minimal CPU threads — let GPU handle it
#     }
    
#     # CRITICAL FIX: Pass token limits and stop words to prevent hallucination loops!
#     if max_tokens:
#         options["num_predict"] = max_tokens
#     if stop:
#         options["stop"] = stop

#     payload = {
#         "model": model,
#         "prompt": prompt,
#         "stream": stream,
#         "options": options,
#     }
    
#     try:
#         resp = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=timeout)
#         resp.raise_for_status()
#         try:
#             data = resp.json()
#             response = data.get("response") or data.get("text")
#         except json.JSONDecodeError:
#             response = resp.text.strip()

#         if not response:
#             raise ValueError("Empty OLLAMA response")

#         elapsed = time.time() - start_time
#         print(f"[OK] OLLAMA: {elapsed:.1f}s\n")
#         return response.strip()

#     except requests.exceptions.Timeout:
#         print(f"\n[ERROR] OLLAMA TIMEOUT after {timeout}s  (model={model})")
#         print(f"   TIP: Ollama may still be loading the model.")
#         print(f"   TIP: If this happens often, raise TIMEOUT_MEDIUM/LONG.\n")
#         return "OLLAMA ERROR"
#     except requests.exceptions.ConnectionError as e:
#         print(f"\n[ERROR] OLLAMA CONNECTION ERROR — is 'ollama serve' running?")
#         print(f"   Details: {str(e)}\n")
#         return "OLLAMA ERROR"
#     except requests.exceptions.HTTPError as e:
#         code = e.response.status_code if hasattr(e, "response") else "?"
#         print(f"\n[ERROR] OLLAMA HTTP {code} — run 'ollama pull {model}'")
#         print(f"   Details: {str(e)}\n")
#         return "OLLAMA ERROR"
#     except Exception as e:
#         print(f"\n[ERROR] OLLAMA {type(e).__name__}: {str(e)}\n")
#         return "OLLAMA ERROR"


# # ── Public single-shot wrappers ──────────────────────────────────────────────

# def ChatGPT_single_request(prompt):
#     temp_sleep()
#     return ollama_request(prompt, model=OLLAMA_CHAT_MODEL, temperature=0,
#                           timeout=TIMEOUT_MEDIUM)


# def GPT4_request(prompt):
#     temp_sleep()
#     try:
#         return ollama_request(prompt, model=OLLAMA_CHAT_MODEL, temperature=0,
#                               timeout=TIMEOUT_MEDIUM)
#     except Exception as e:
#         print(f"[ERROR] GPT4_request: {type(e).__name__}: {e}")
#         return "ChatGPT ERROR"


# def ChatGPT_request(prompt):
#     try:
#         return ollama_request(prompt, model=OLLAMA_CHAT_MODEL, temperature=0,
#                               timeout=TIMEOUT_MEDIUM)
#     except Exception as e:
#         print(f"[ERROR] ChatGPT_request: {type(e).__name__}: {e}")
#         return "ChatGPT ERROR"


# # ── Safe generate helpers ────────────────────────────────────────────────────

# def _is_error_response(text: str) -> bool:
#     """Returns True if the response is an Ollama/token error sentinel."""
#     return ("OLLAMA ERROR" in text
#             or "TOKEN LIMIT" in text
#             or "ChatGPT ERROR" in text)


# def GPT4_safe_generate_response(prompt,
#                                 example_output,
#                                 special_instruction,
#                                 repeat=3,
#                                 fail_safe_response="error",
#                                 func_validate=None,
#                                 func_clean_up=None,
#                                 verbose=False):
#     prompt = '"""\n' + prompt + '\n"""\n'
#     prompt += (f"Output the response to the prompt above in json. "
#                f"{special_instruction}\n")
#     prompt += "Example output json:\n"
#     prompt += '{"output": "' + str(example_output) + '"}'

#     for i in range(repeat):
#         try:
#             curr = GPT4_request(prompt).strip()
#             if _is_error_response(curr):
#                 continue
#             # Try JSON extraction, fall back to plain text
#             try:
#                 end = curr.rfind("}") + 1
#                 if end > 0:
#                     curr = json.loads(curr[:end])["output"]
#             except (json.JSONDecodeError, KeyError):
#                 pass
#             if func_validate(curr, prompt=prompt):
#                 return func_clean_up(curr, prompt=prompt)
#             if verbose:
#                 print(f"[GPT4_safe attempt {i+1}] validation failed: {curr[:80]}")
#         except Exception as e:
#             if verbose:
#                 print(f"[GPT4_safe attempt {i+1}] {type(e).__name__}: {e}")
#     return False


# def ChatGPT_safe_generate_response(prompt,
#                                    example_output,
#                                    special_instruction,
#                                    repeat=3,
#                                    fail_safe_response="error",
#                                    func_validate=None,
#                                    func_clean_up=None,
#                                    verbose=False):
#     prompt = '"""\n' + prompt + '\n"""\n'
#     prompt += (f"Output the response to the prompt above in json. "
#                f"{special_instruction}\n")
#     prompt += "Example output json:\n"
#     prompt += '{"output": "' + str(example_output) + '"}'

#     for i in range(repeat):
#         try:
#             curr = ChatGPT_request(prompt).strip()
#             if _is_error_response(curr):
#                 continue
#             try:
#                 end = curr.rfind("}") + 1
#                 if end > 0:
#                     curr = json.loads(curr[:end])["output"]
#             except (json.JSONDecodeError, KeyError):
#                 pass
#             if func_validate(curr, prompt=prompt):
#                 return func_clean_up(curr, prompt=prompt)
#             if verbose:
#                 print(f"[ChatGPT_safe attempt {i+1}] validation failed: {curr[:80]}")
#         except Exception as e:
#             if verbose:
#                 print(f"[ChatGPT_safe attempt {i+1}] {type(e).__name__}: {e}")
#     return fail_safe_response


# def ChatGPT_safe_generate_response_OLD(prompt,
#                                        repeat=3,
#                                        fail_safe_response="error",
#                                        func_validate=None,
#                                        func_clean_up=None,
#                                        verbose=False):
#     for i in range(repeat):
#         try:
#             curr = ChatGPT_request(prompt).strip()
#             if _is_error_response(curr):
#                 continue
#             if func_validate(curr, prompt=prompt):
#                 return func_clean_up(curr, prompt=prompt)
#             if verbose:
#                 print(f"[ChatGPT_safe_OLD attempt {i+1}] validation failed: {curr[:80]}")
#         except Exception as e:
#             if verbose:
#                 print(f"[ChatGPT_safe_OLD attempt {i+1}] {type(e).__name__}: {e}")
#     print("FAIL SAFE TRIGGERED")
#     return fail_safe_response


# # ── Original GPT-3 completion path ──────────────────────────────────────────

# def GPT_request(prompt, gpt_parameter):
#     """
#     Maps the legacy OpenAI API call to Ollama.
#     Uses TIMEOUT_SHORT for low max_tokens requests (≤ 50 tokens) and
#     TIMEOUT_MEDIUM otherwise, so simple prompts fail fast.
#     """
#     temp_sleep()
#     try:
#         temperature = gpt_parameter.get("temperature", 0)
#         max_tokens  = gpt_parameter.get("max_tokens", 200)
#         stop = gpt_parameter.get("stop", None) # Retrieve the stop commands!
        
#         # Short completions (pronunciatio, wake_up_hour, etc.) get a tighter
#         # timeout so the simulation doesn't stall on trivial prompts.
#         timeout = TIMEOUT_SHORT if max_tokens <= 50 else TIMEOUT_MEDIUM
        
#         response = ollama_request(prompt, model=OLLAMA_CHAT_MODEL,
#                                   temperature=temperature, timeout=timeout,
#                                   max_tokens=max_tokens, stop=stop)
        
#         if _is_error_response(response):
#             print("[ERROR] OLLAMA ERROR detected in GPT_request response")
#             return "OLLAMA ERROR"
#         return response
#     except Exception as e:
#         print(f"[ERROR] GPT_request EXCEPTION: {type(e).__name__}: {e}")
#         return "TOKEN LIMIT EXCEEDED"


# def generate_prompt(curr_input, prompt_lib_file):
#     """
#     Fills !<INPUT N>! placeholders in a prompt template file.
#     """
#     if isinstance(curr_input, str):
#         curr_input = [curr_input]
#     curr_input = [str(i) for i in curr_input]

#     with open(prompt_lib_file, "r") as f:
#         prompt = f.read()
#     for count, i in enumerate(curr_input):
#         prompt = prompt.replace(f"!<INPUT {count}>!", i)
#     if "<commentblockmarker>###</commentblockmarker>" in prompt:
#         prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
#     return prompt.strip()


# def safe_generate_response(prompt,
#                            gpt_parameter,
#                            repeat=5,
#                            fail_safe_response="error",
#                            func_validate=None,
#                            func_clean_up=None,
#                            verbose=False):
#     """
#     Calls GPT_request up to `repeat` times, validating each response.

#     Changes vs original:
#     - Error responses skip validation immediately (no wasted retry time).
#     - repeat capped at 3 for short-token prompts to save time.
#     - Fail-safe is returned (not raised) after all attempts exhausted.
#     """
#     if verbose:
#         print(prompt)

#     # Cap retries for simple prompts — fail fast and use the fail-safe.
#     max_tokens = gpt_parameter.get("max_tokens", 200)
#     effective_repeat = min(repeat, 3) if max_tokens <= 50 else repeat

#     for i in range(effective_repeat):
#         try:
#             curr_gpt_response = GPT_request(prompt, gpt_parameter)

#             if _is_error_response(curr_gpt_response):
#                 print(f"[Attempt {i+1}/{effective_repeat}] Error response — "
#                       f"using fail-safe on next pass" if i == effective_repeat - 1
#                       else f"[Attempt {i+1}/{effective_repeat}] Error — retrying…")
#                 continue

#             if func_validate(curr_gpt_response, prompt=prompt):
#                 return func_clean_up(curr_gpt_response, prompt=prompt)

#             if verbose:
#                 print(f"[Attempt {i+1}/{effective_repeat}] Validation failed:")
#                 print(curr_gpt_response[:200])
#         except Exception as e:
#             print(f"[Attempt {i+1}/{effective_repeat}] Exception: "
#                   f"{type(e).__name__}: {str(e)[:100]}")
#             continue

#     print(f"[WARNING] safe_generate_response gave up after "
#           f"{effective_repeat} attempts — returning fail_safe")
#     return fail_safe_response


# # ── Embeddings ───────────────────────────────────────────────────────────────

# def get_embedding(text, model="text-embedding-ada-002"):
#     """
#     Returns a float vector from the local Ollama embedding model.
#     Falls back through two known Ollama endpoint paths.
#     Returns "EMBEDDING ERROR" string on failure — callers must handle this.
#     """
#     text = text.replace("\n", " ").strip()
#     if not text:
#         text = "this is blank"

#     # Keep embedding prompts short — long texts slow down nomic-embed-text.
#     if len(text) > 2000:
#         text = text[:2000]

#     OLLAMA_EMBED_URLS = [
#         OLLAMA_BASE_URL + "/api/embeddings",
#         OLLAMA_BASE_URL + "/api/embed",
#     ]

#     last_exc = None
#     for url in OLLAMA_EMBED_URLS:
#         try:
#             resp = requests.post(
#                 url,
#                 json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
#                 timeout=TIMEOUT_EMBED,
#             )
#             resp.raise_for_status()
#             data = resp.json()
#             if isinstance(data, dict):
#                 if "embedding" in data:
#                     return data["embedding"]
#                 if ("data" in data
#                         and isinstance(data["data"], list)
#                         and "embedding" in data["data"][0]):
#                     return data["data"][0]["embedding"]
#             return data
#         except Exception as e:
#             last_exc = e
#             continue

#     print(f"[ERROR] EMBEDDING ERROR: {type(last_exc).__name__}: {last_exc}")
#     return "EMBEDDING ERROR"


# if __name__ == "__main__":
#     gpt_parameter = {
#         "engine": "text-davinci-003", "max_tokens": 50,
#         "temperature": 0, "top_p": 1, "stream": False,
#         "frequency_penalty": 0, "presence_penalty": 0, "stop": ['"'],
#     }
#     curr_input = ["driving to a friend's house"]
#     prompt_lib_file = "prompt_template/test_prompt_July5.txt"
#     prompt = generate_prompt(curr_input, prompt_lib_file)

#     def __func_validate(gpt_response, prompt=""):
#         if len(gpt_response.strip()) <= 1:
#             return False
#         if len(gpt_response.strip().split(" ")) > 1:
#             return False
#         return True

#     def __func_clean_up(gpt_response, prompt=""):
#         return gpt_response.strip()

#     output = safe_generate_response(
#         prompt, gpt_parameter, 5, "rest",
#         __func_validate, __func_clean_up, True
#     )
#     print(output)