import os
import pathlib
import traceback
import networkx as nx
import gamms

from lib.core import *
import lib.game_utils as GMUTL
import lib.time_logger as TLOG
from typing import Optional, Tuple


# --- Main Game Runner ---
def run_game(config_name: str, root_dir: str, attacker_strategy, defender_strategy,
             logger: Optional[TLOG.TimeLogger] = None, visualization: bool = False, debug: bool = False) -> Tuple[float, int, int, int]:
    """
    Runs the multi-agent game using the provided configuration.

    Returns:
        tuple: (final_payoff, time, total_captures, total_tags)
    """
    # Initialize variables outside the try to ensure they exist in finally block
    time_counter = 0
    payoff = 0
    total_tags = 0
    total_captures = 0
    ctx = None
    error_message = None
    
    try:
        dirs = GMUTL.get_directories(root_dir)
        config = GMUTL.load_configuration(config_name, dirs, debug)
        G = GMUTL.load_graph(config, dirs, debug)
        
        # Create static sensor definitions once
        static_sensors = GMUTL.create_static_sensors()
        
        # Create a new context with sensors for this run
        ctx = GMUTL.create_context_with_sensors(config, G, visualization, static_sensors, debug)
        
        # Initialize agents and assign strategies
        agent_config, agent_params_dict = GMUTL.initialize_agents(ctx, config)
        GMUTL.assign_strategies(ctx, agent_config, attacker_strategy, defender_strategy)
        success("Assigned strategies successfully", debug)
        
        # Configure visualization and initialize flags
        GMUTL.configure_visualization(ctx, agent_config, config)
        GMUTL.initialize_flags(ctx, config)
        
        # Retrieve game parameters
        max_time = config.get("game", {}).get("max_time", 1000)
        flag_positions = config.get("game", {}).get("flag", {}).get("positions", [])
        flag_weights = config.get("game", {}).get("flag", {}).get("weights")
        interaction_config = config.get("game", {}).get("interaction", {})
        payoff_config = config.get("game", {}).get("payoff", {})

        # Initialize the time logger
        metadata = GMUTL.load_config_metadata(config)
        if logger is not None:
            current_metadata = logger.get_metadata()
            merged_metadata = {**current_metadata, **metadata}
            logger.set_metadata(merged_metadata)
        success("Initialized time logger", debug)
        
        success(f"Starting game with max time: {max_time}", debug)
        
        # Check initial interactions before any moves
        init_caps, init_tags, attacker_count, defender_count, init_cap_details, init_tag_details = GMUTL.check_agent_interaction(
            ctx, G, agent_params_dict, flag_positions, interaction_config, time_counter
        )
        total_captures += init_caps
        total_tags += init_tags
        payoff += GMUTL.compute_payoff(payoff_config, init_caps, init_tags)
        
        # Log initial state
        initial_step_log = {
            "agents": {agent.name: agent.get_state().get("curr_pos") for agent in ctx.agent.create_iter()},
            "flag_positions": flag_positions,
            "payoff": payoff,
            "captures": init_caps,
            "tags": init_tags,
            "tag_details": init_tag_details,
            "capture_details": init_cap_details,
        }
        if logger is not None:
            logger.log_data(initial_step_log, time_counter)
        
        # Check if game should terminate after initial state
        if GMUTL.check_termination(time_counter, max_time, attacker_count, defender_count):
            success(f"Game terminated at time {time_counter}", debug)
            return payoff, time_counter, total_captures, total_tags

        # Main game loop
        while not ctx.is_terminated():
            time_counter += 1
            next_actions = {}
            
            # Compute next actions for all agents
            try:
                for agent in ctx.agent.create_iter():
                    state = agent.get_state()
                    state.update({
                        "flag_pos": flag_positions,
                        "flag_weight": flag_weights,
                        "agent_params": agent_params_dict.get(agent.name, {}),
                        "time": time_counter,
                        "payoff": payoff,
                        "name": agent.name,
                        "agent_params_dict": agent_params_dict
                    })
                    if hasattr(agent, "strategy") and agent.strategy is not None:
                        try:
                            agent.strategy(state)
                        except Exception as e:
                            error(f"Error executing strategy for {agent.name}: {e}")
                            traceback.print_exc()
                    else:
                        node = ctx.visual.human_input(agent.name, state)
                        state["action"] = node
                    next_actions[agent.name] = state["action"]
            except Exception as e:
                error(f"An error occurred during agent turn: {e}")
                raise e
            
            # Update agents with their actions
            for agent in ctx.agent.create_iter():
                state = agent.get_state()
                state["action"] = next_actions.get(agent.name, state.get("action", None))
                agent.set_state()
            
            # Update visualization display (simulate handles pygame events internally)
            ctx.visual.simulate()
            
            # Log current agent positions (using "curr_pos" from state)
            agent_positions = {agent.name: agent.get_state().get("curr_pos") for agent in ctx.agent.create_iter()}
            
            # Check interactions (captures, tags, etc.)
            captures, tags, attacker_count, defender_count, capture_details, tag_details = GMUTL.check_agent_interaction(
                ctx, G, agent_params_dict, flag_positions, interaction_config, time_counter
            )
            total_captures += captures
            total_tags += tags
            payoff += GMUTL.compute_payoff(payoff_config, captures, tags)
            
            step_log = {
                "agents": agent_positions,
                "flag_positions": flag_positions,
                "payoff": payoff,
                "captures": captures,
                "tags": tags,
                "tag_details": tag_details,
                "capture_details": capture_details,
            }
            if logger is not None:
                logger.log_data(step_log, time_counter)
            
            if GMUTL.check_termination(time_counter, max_time, attacker_count, defender_count):
                success(f"Game terminated at time {time_counter}", debug)
                break
        
        return payoff, time_counter, total_captures, total_tags

    except KeyboardInterrupt:
        warning("Game interrupted by user.")
        error_message = "Game interrupted by user"
        return payoff, time_counter, total_captures, total_tags
    except Exception as e:
        error_msg = f"An error occurred: {e}"
        error(error_msg)
        error_message = f"{error_msg}\n{traceback.format_exc()}"
        traceback.print_exc()
        return payoff, time_counter, total_captures, total_tags
    finally:
        # Always finalize the logger if it exists
        if logger is not None:
            # Include error information in the finalization if an error occurred
            if error_message:
                logger.finalize(
                    payoff=payoff, 
                    time=time_counter, 
                    total_captures=total_captures, 
                    total_tags=total_tags,
                    error=error_message
                )
            else:
                logger.finalize(
                    payoff=payoff, 
                    time=time_counter, 
                    total_captures=total_captures, 
                    total_tags=total_tags
                )
        success("Game completed", debug)


if __name__ == "__main__":
    current_path = pathlib.Path(__file__).resolve()
    root_path = current_path.parent
    print("Current path:", current_path)
    print("Root path:", root_path)

    import policies.attacker.GMU_Attacker as attacker
    import policies.defender.UNCC_Defender as defender

    RESULT_PATH = os.path.join(root_path, "data/result")
    logger = TLOG.TimeLogger("test", path=RESULT_PATH)

    final_payoff, game_time, _, _ = run_game("config_a47dfbcc6f.yml", root_dir=str(root_path), attacker_strategy=attacker, defender_strategy=defender, logger=logger, visualization=False, debug=True)
    # logger.write_to_file("test.json", force=True)
    print("Final payoff:", final_payoff)
