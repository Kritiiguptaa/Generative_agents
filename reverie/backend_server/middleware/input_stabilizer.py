# middleware/input_stabilizer.py

from middleware.config.middleware_config import MiddlewareConfig
from middleware.semantic_density_gating import SemanticDensityGating
from middleware.memory_compression import MemoryCompression
from middleware.persona_anchoring import PersonaAnchoring
from middleware.action_filtering import ActionFiltering
from typing import Dict, Tuple


class InputStabilizer:
    """
    ORCHESTRATOR: 4-step pipeline to clean, compress, and anchor agent memories.
    
    This class coordinates all 4 steps:
    1. SemanticDensityGating - Filter out low-entropy memories
    2. MemoryCompression - Compress filtered memories
    3. PersonaAnchoring - Create identity block
    4. ActionFiltering - Whitelist legal actions
    """
    
    def __init__(self, config: MiddlewareConfig):
        """
        Initialize stabilizer with all 4 sub-components.
        
        Args:
            config (MiddlewareConfig): Configuration object
        """
        self.config = config
        self.density_gating = SemanticDensityGating(config)
        self.compression = MemoryCompression()
        self.anchoring = PersonaAnchoring(config)
        self.action_filter = ActionFiltering()
    
    def stabilize_input(
        self,
        persona,
        retrieved_memories: Dict,
        current_prompt: str,
        maze
    ) -> Tuple[Dict, str]:
        """
        Main entry point: orchestrate all 4 steps.
        
        Args:
            persona: Agent persona object
            retrieved_memories: Dict of memories from persona.retrieve()
            current_prompt: The base prompt to send to LLM
            maze: The game maze for validating legal actions
            
        Returns:
            Tuple of (stabilized_context_dict, embedded_anchor_string)
        """
        # STEP 1: Filter low-entropy memories
        filtered_memories = self.density_gating.filter_low_entropy_memories(
            retrieved_memories.get("events", [])
        )
        
        # STEP 2: Compress into structured object
        compressed = self.compression.compress_memories(filtered_memories, persona)
        
        # STEP 3: Create persona anchor
        anchor = self.anchoring.create_persona_anchor(persona)
        
        # STEP 4: Get legal actions
        legal_actions = self.action_filter.restrict_to_legal_actions(maze, persona)
        
        # Build stabilized context
        stabilized_context = {
            "compressed_memories": compressed,
            "legal_actions": legal_actions,
            "anchor": anchor
        }
        
        # Embed anchor into prompt
        embedded_prompt = f"{anchor}\n\n{current_prompt}"
        
        return stabilized_context, embedded_prompt