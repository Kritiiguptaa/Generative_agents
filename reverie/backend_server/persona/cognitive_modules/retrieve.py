# """
# Author: Joon Sung Park (joonspk@stanford.edu)

# File: retrieve.py
# Description: This defines the "Retrieve" module for generative agents. 
# """
# import sys
# sys.path.append('../../')

# from global_methods import *
# from persona.prompt_template.gpt_structure import *

# from numpy import dot
# from numpy.linalg import norm

# def retrieve(persona, perceived): 
#   """
#   This function takes the events that are perceived by the persona as input
#   and returns a set of related events and thoughts that the persona would 
#   need to consider as context when planning. 

#   INPUT: 
#     perceived: a list of event <ConceptNode>s that represent any of the events
#     `         that are happening around the persona. What is included in here
#               are controlled by the att_bandwidth and retention 
#               hyper-parameters.
#   OUTPUT: 
#     retrieved: a dictionary of dictionary. The first layer specifies an event, 
#                while the latter layer specifies the "curr_event", "events", 
#                and "thoughts" that are relevant.
#   """
#   # We rerieve events and thoughts separately. 
#   retrieved = dict()
#   for event in perceived: 
#     retrieved[event.description] = dict()
#     retrieved[event.description]["curr_event"] = event
    
#     relevant_events = persona.a_mem.retrieve_relevant_events(
#                         event.subject, event.predicate, event.object)
#     retrieved[event.description]["events"] = list(relevant_events)

#     relevant_thoughts = persona.a_mem.retrieve_relevant_thoughts(
#                           event.subject, event.predicate, event.object)
#     retrieved[event.description]["thoughts"] = list(relevant_thoughts)
    
#   return retrieved


# def cos_sim(a, b): 
#   """
#   This function calculates the cosine similarity between two input vectors 
#   'a' and 'b'. Cosine similarity is a measure of similarity between two 
#   non-zero vectors of an inner product space that measures the cosine 
#   of the angle between them.

#   INPUT: 
#     a: 1-D array object 
#     b: 1-D array object 
#   OUTPUT: 
#     A scalar value representing the cosine similarity between the input 
#     vectors 'a' and 'b'.
  
#   Example input: 
#     a = [0.3, 0.2, 0.5]
#     b = [0.2, 0.2, 0.5]
#   """
#   return dot(a, b)/(norm(a)*norm(b))


# def normalize_dict_floats(d, target_min, target_max):
#   """
#   This function normalizes the float values of a given dictionary 'd' between 
#   a target minimum and maximum value. The normalization is done by scaling the
#   values to the target range while maintaining the same relative proportions 
#   between the original values.

#   INPUT: 
#     d: Dictionary. The input dictionary whose float values need to be 
#        normalized.
#     target_min: Integer or float. The minimum value to which the original 
#                 values should be scaled.
#     target_max: Integer or float. The maximum value to which the original 
#                 values should be scaled.
#   OUTPUT: 
#     d: A new dictionary with the same keys as the input but with the float
#        values normalized between the target_min and target_max.

#   Example input: 
#     d = {'a':1.2,'b':3.4,'c':5.6,'d':7.8}
#     target_min = -5
#     target_max = 5
#   """
#   min_val = min(val for val in d.values())
#   max_val = max(val for val in d.values())
#   range_val = max_val - min_val

#   if range_val == 0: 
#     for key, val in d.items(): 
#       d[key] = (target_max - target_min)/2
#   else: 
#     for key, val in d.items():
#       d[key] = ((val - min_val) * (target_max - target_min) 
#                 / range_val + target_min)
#   return d


# def top_highest_x_values(d, x):
#   """
#   This function takes a dictionary 'd' and an integer 'x' as input, and 
#   returns a new dictionary containing the top 'x' key-value pairs from the 
#   input dictionary 'd' with the highest values.

#   INPUT: 
#     d: Dictionary. The input dictionary from which the top 'x' key-value pairs 
#        with the highest values are to be extracted.
#     x: Integer. The number of top key-value pairs with the highest values to
#        be extracted from the input dictionary.
#   OUTPUT: 
#     A new dictionary containing the top 'x' key-value pairs from the input 
#     dictionary 'd' with the highest values.
  
#   Example input: 
#     d = {'a':1.2,'b':3.4,'c':5.6,'d':7.8}
#     x = 3
#   """
#   top_v = dict(sorted(d.items(), 
#                       key=lambda item: item[1], 
#                       reverse=True)[:x])
#   return top_v


# def extract_recency(persona, nodes):
#   """
#   Gets the current Persona object and a list of nodes that are in a 
#   chronological order, and outputs a dictionary that has the recency score
#   calculated.

#   INPUT: 
#     persona: Current persona whose memory we are retrieving. 
#     nodes: A list of Node object in a chronological order. 
#   OUTPUT: 
#     recency_out: A dictionary whose keys are the node.node_id and whose values
#                  are the float that represents the recency score. 
#   """
#   recency_vals = [persona.scratch.recency_decay ** i 
#                   for i in range(1, len(nodes) + 1)]
  
#   recency_out = dict()
#   for count, node in enumerate(nodes): 
#     recency_out[node.node_id] = recency_vals[count]

#   return recency_out


# def extract_importance(persona, nodes):
#   """
#   Gets the current Persona object and a list of nodes that are in a 
#   chronological order, and outputs a dictionary that has the importance score
#   calculated.

#   INPUT: 
#     persona: Current persona whose memory we are retrieving. 
#     nodes: A list of Node object in a chronological order. 
#   OUTPUT: 
#     importance_out: A dictionary whose keys are the node.node_id and whose 
#                     values are the float that represents the importance score.
#   """
#   importance_out = dict()
#   for count, node in enumerate(nodes): 
#     importance_out[node.node_id] = node.poignancy

#   return importance_out


# def extract_relevance(persona, nodes, focal_pt): 
#   """
#   Gets the current Persona object, a list of nodes that are in a 
#   chronological order, and the focal_pt string and outputs a dictionary 
#   that has the relevance score calculated.

#   INPUT: 
#     persona: Current persona whose memory we are retrieving. 
#     nodes: A list of Node object in a chronological order. 
#     focal_pt: A string describing the current thought of revent of focus.  
#   OUTPUT: 
#     relevance_out: A dictionary whose keys are the node.node_id and whose values
#                  are the float that represents the relevance score. 
#   """
#   focal_embedding = get_embedding(focal_pt)

#   relevance_out = dict()
#   for count, node in enumerate(nodes): 
#     node_embedding = persona.a_mem.embeddings[node.embedding_key]
#     relevance_out[node.node_id] = cos_sim(node_embedding, focal_embedding)

#   return relevance_out


# def new_retrieve(persona, focal_points, n_count=30): 
#   """
#   Given the current persona and focal points (focal points are events or 
#   thoughts for which we are retrieving), we retrieve a set of nodes for each
#   of the focal points and return a dictionary. 

#   INPUT: 
#     persona: The current persona object whose memory we are retrieving. 
#     focal_points: A list of focal points (string description of the events or
#                   thoughts that is the focus of current retrieval).
#   OUTPUT: 
#     retrieved: A dictionary whose keys are a string focal point, and whose 
#                values are a list of Node object in the agent's associative 
#                memory.

#   Example input:
#     persona = <persona> object 
#     focal_points = ["How are you?", "Jane is swimming in the pond"]
#   """
#   # <retrieved> is the main dictionary that we are returning
#   retrieved = dict() 
#   for focal_pt in focal_points: 
#     # Getting all nodes from the agent's memory (both thoughts and events) and
#     # sorting them by the datetime of creation.
#     # You could also imagine getting the raw conversation, but for now. 
#     nodes = [[i.last_accessed, i]
#               for i in persona.a_mem.seq_event + persona.a_mem.seq_thought
#               if "idle" not in i.embedding_key]
#     nodes = sorted(nodes, key=lambda x: x[0])
#     nodes = [i for created, i in nodes]

#     # Calculating the component dictionaries and normalizing them.
#     recency_out = extract_recency(persona, nodes)
#     recency_out = normalize_dict_floats(recency_out, 0, 1)
#     importance_out = extract_importance(persona, nodes)
#     importance_out = normalize_dict_floats(importance_out, 0, 1)  
#     relevance_out = extract_relevance(persona, nodes, focal_pt)
#     relevance_out = normalize_dict_floats(relevance_out, 0, 1)

#     # Computing the final scores that combines the component values. 
#     # Note to self: test out different weights. [1, 1, 1] tends to work
#     # decently, but in the future, these weights should likely be learned, 
#     # perhaps through an RL-like process.
#     # gw = [1, 1, 1]
#     # gw = [1, 2, 1]
#     gw = [0.5, 3, 2]
#     master_out = dict()
#     for key in recency_out.keys(): 
#       master_out[key] = (persona.scratch.recency_w*recency_out[key]*gw[0] 
#                      + persona.scratch.relevance_w*relevance_out[key]*gw[1] 
#                      + persona.scratch.importance_w*importance_out[key]*gw[2])

#     master_out = top_highest_x_values(master_out, len(master_out.keys()))
#     for key, val in master_out.items(): 
#       print (persona.a_mem.id_to_node[key].embedding_key, val)
#       print (persona.scratch.recency_w*recency_out[key]*1, 
#              persona.scratch.relevance_w*relevance_out[key]*1, 
#              persona.scratch.importance_w*importance_out[key]*1)

#     # Extracting the highest x values.
#     # <master_out> has the key of node.id and value of float. Once we get the 
#     # highest x values, we want to translate the node.id into nodes and return
#     # the list of nodes.
#     master_out = top_highest_x_values(master_out, n_count)
#     master_nodes = [persona.a_mem.id_to_node[key] 
#                     for key in list(master_out.keys())]

#     for n in master_nodes: 
#       n.last_accessed = persona.scratch.curr_time
      
#     retrieved[focal_pt] = master_nodes

#   return retrieved






"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: retrieve.py
Description: This defines the "Retrieve" module for generative agents. 
"""
import sys
sys.path.append('../../')

from global_methods import *
from persona.prompt_template.gpt_structure import *

from numpy import dot, array
from numpy.linalg import norm


def retrieve(persona, perceived): 
  """
  This function takes the events that are perceived by the persona as input
  and returns a set of related events and thoughts that the persona would 
  need to consider as context when planning. 

  INPUT: 
    perceived: a list of event <ConceptNode>s that represent any of the events
               that are happening around the persona. What is included in here
               are controlled by the att_bandwidth and retention 
               hyper-parameters.
  OUTPUT: 
    retrieved: a dictionary of dictionary. The first layer specifies an event, 
               while the latter layer specifies the "curr_event", "events", 
               and "thoughts" that are relevant.
  """
  retrieved = dict()
  for event in perceived: 
    retrieved[event.description] = dict()
    retrieved[event.description]["curr_event"] = event
    
    relevant_events = persona.a_mem.retrieve_relevant_events(
                        event.subject, event.predicate, event.object)
    retrieved[event.description]["events"] = list(relevant_events)

    relevant_thoughts = persona.a_mem.retrieve_relevant_thoughts(
                          event.subject, event.predicate, event.object)
    retrieved[event.description]["thoughts"] = list(relevant_thoughts)
    
  return retrieved


def cos_sim(a, b): 
  """
  Calculates cosine similarity between two vectors.
  Both inputs must be 1-D array-like objects of floats.

  Returns 0.0 safely if either vector has zero norm (avoids division by zero).
  """
  try:
    a = array(a, dtype=float)
    b = array(b, dtype=float)
    denom = norm(a) * norm(b)
    if denom == 0:
      return 0.0
    return float(dot(a, b) / denom)
  except Exception:
    return 0.0


def normalize_dict_floats(d, target_min, target_max):
  """
  Normalizes the float values of a dictionary between target_min and target_max.
  """
  if not d:
    return d

  min_val = min(val for val in d.values())
  max_val = max(val for val in d.values())
  range_val = max_val - min_val

  if range_val == 0: 
    for key in d: 
      d[key] = (target_max - target_min) / 2
  else: 
    for key, val in d.items():
      d[key] = ((val - min_val) * (target_max - target_min) 
                / range_val + target_min)
  return d


def top_highest_x_values(d, x):
  """
  Returns a new dict with the top x key-value pairs by highest value.
  """
  top_v = dict(sorted(d.items(), 
                      key=lambda item: item[1], 
                      reverse=True)[:x])
  return top_v


def extract_recency(persona, nodes):
  """
  Calculates recency scores for a list of nodes using exponential decay.

  OUTPUT:
    recency_out: dict mapping node.node_id -> float recency score
  """
  recency_vals = [persona.scratch.recency_decay ** i 
                  for i in range(1, len(nodes) + 1)]
  
  recency_out = dict()
  for count, node in enumerate(nodes): 
    recency_out[node.node_id] = recency_vals[count]

  return recency_out


def extract_importance(persona, nodes):
  """
  Returns poignancy scores for a list of nodes.

  OUTPUT:
    importance_out: dict mapping node.node_id -> float importance score
  """
  importance_out = dict()
  for count, node in enumerate(nodes): 
    importance_out[node.node_id] = node.poignancy

  return importance_out


def _safe_get_embedding(persona, node):
  """
  Safely retrieves the embedding vector for a node from persona.a_mem.embeddings.

  Handles three failure modes:
    1. Key is missing from embeddings dict  -> calls get_embedding() to regenerate
    2. Stored value is a dict (corrupted)   -> calls get_embedding() to regenerate
    3. get_embedding() itself fails         -> returns None

  When regenerating, the fresh embedding is stored back into a_mem.embeddings
  so the fix persists for the rest of the session (and gets saved on next
  persona.save() call).
  """
  raw = persona.a_mem.embeddings.get(node.embedding_key, None)

  # Valid case: raw is a list of floats
  if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], (int, float)):
    return raw

  # Invalid case: missing or wrong type — regenerate from the node description
  print(f"[retrieve] WARNING: embedding for key '{node.embedding_key}' is "
        f"{'missing' if raw is None else 'malformed (' + type(raw).__name__ + ')'}. "
        f"Regenerating from description: '{node.description[:60]}'...")
  try:
    fresh = get_embedding(node.description)
    if isinstance(fresh, list) and len(fresh) > 0:
      persona.a_mem.embeddings[node.embedding_key] = fresh
      return fresh
  except Exception as e:
    print(f"[retrieve] ERROR: Could not regenerate embedding for "
          f"'{node.embedding_key}': {e}")

  return None


def extract_relevance(persona, nodes, focal_pt): 
  """
  Calculates cosine similarity between each node's embedding and the
  focal point embedding.

  Safely handles missing or malformed embeddings — nodes whose embeddings
  cannot be retrieved or regenerated are assigned a relevance score of 0.0
  rather than crashing the server.

  INPUT: 
    persona:   Current persona whose memory we are retrieving. 
    nodes:     A list of Node objects in chronological order. 
    focal_pt:  A string describing the current thought or event of focus.  
  OUTPUT: 
    relevance_out: dict mapping node.node_id -> float relevance score
  """
  focal_embedding = get_embedding(focal_pt)

  relevance_out = dict()
  for node in nodes:
    node_embedding = _safe_get_embedding(persona, node)

    if node_embedding is None:
      # Can't compute similarity — assign neutral score so this node
      # can still be ranked (it won't be top-ranked but won't crash)
      relevance_out[node.node_id] = 0.0
    else:
      relevance_out[node.node_id] = cos_sim(node_embedding, focal_embedding)

  return relevance_out


def new_retrieve(persona, focal_points, n_count=30): 
  """
  Given the current persona and focal points, retrieves the most relevant
  nodes from associative memory for each focal point.

  INPUT: 
    persona:       The current persona object whose memory we are retrieving. 
    focal_points:  A list of strings describing the events/thoughts in focus.
    n_count:       Number of top nodes to return per focal point.
  OUTPUT: 
    retrieved: dict mapping focal_point string -> list of top Node objects
  """
  retrieved = dict() 
  for focal_pt in focal_points: 
    # Collect all non-idle nodes, sorted by last_accessed time
    nodes = [[i.last_accessed, i]
              for i in persona.a_mem.seq_event + persona.a_mem.seq_thought
              if "idle" not in i.embedding_key]
    nodes = sorted(nodes, key=lambda x: x[0])
    nodes = [i for created, i in nodes]

    if not nodes:
      retrieved[focal_pt] = []
      continue

    # Calculating component scores
    recency_out    = extract_recency(persona, nodes)
    recency_out    = normalize_dict_floats(recency_out, 0, 1)
    importance_out = extract_importance(persona, nodes)
    importance_out = normalize_dict_floats(importance_out, 0, 1)  
    relevance_out  = extract_relevance(persona, nodes, focal_pt)
    relevance_out  = normalize_dict_floats(relevance_out, 0, 1)

    # Weighted combination: [recency_w=0.5, relevance_w=3, importance_w=2]
    gw = [0.5, 3, 2]
    master_out = dict()
    for key in recency_out.keys(): 
      master_out[key] = (persona.scratch.recency_w    * recency_out[key]    * gw[0]
                       + persona.scratch.relevance_w  * relevance_out[key]  * gw[1] 
                       + persona.scratch.importance_w * importance_out[key] * gw[2])

    # Select top n_count nodes
    master_out = top_highest_x_values(master_out, n_count)
    master_nodes = [persona.a_mem.id_to_node[key] 
                    for key in list(master_out.keys())]

    for n in master_nodes: 
      n.last_accessed = persona.scratch.curr_time
      
    retrieved[focal_pt] = master_nodes

  return retrieved