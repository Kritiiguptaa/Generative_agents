# #!/usr/bin/env python3
# """
# Quick test script to run simulation without interactive prompts
# """
# import sys
# sys.path.append('.')

# from reverie import ReverieServer

# # Create simulation with origin and target
# import time
# origin = "base_test_hallucination"
# target = f"test_verified_{int(time.time())}"

# print(f"Starting simulation: {origin} -> {target}")
# rs = ReverieServer(origin, target)
# print(f"[OK] ReverieServer created")

# # Run 5 steps
# try:
#     print(f"Running 5 simulation steps...")
#     rs.start_server(5)
#     print(f"[OK] Simulation completed 5 steps successfully!")
# except Exception as e:
#     print(f"[ERROR] Error during simulation: {type(e).__name__}: {str(e)}")
#     import traceback
#     traceback.print_exc()



#!/usr/bin/env python3
"""
Quick test script to run simulation without interactive prompts
"""
import sys
sys.path.append('.')

from reverie import ReverieServer

import time
origin = "base_test_hallucination"
target = f"test_verified_{int(time.time())}"

print(f"Starting simulation: {origin} -> {target}")
rs = ReverieServer(origin, target)
print(f"[OK] ReverieServer created")

try:
    print(f"Running 5 simulation steps...")
    rs.start_server(5)
    print(f"[OK] Simulation completed 5 steps successfully!")
except Exception as e:
    print(f"[ERROR] Error during simulation: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
finally:
    # Save always runs — even if start_server() raised an exception.
    # This ensures embeddings.json, nodes.json, and scratch.json are
    # written to disk so memory is not lost on exit.
    print(f"Saving persona memory to disk...")
    try:
        rs.save()
        print(f"[OK] Saved -> environment/frontend_server/storage/{target}")
    except Exception as save_err:
        print(f"[ERROR] Save failed: {type(save_err).__name__}: {str(save_err)}")
        import traceback
        traceback.print_exc()
