from typing import Any, Dict, List, Optional, Tuple
from typeguard import typechecked
import datetime
import hashlib
import gamms
import yaml
import os

try:
    from lib.core import *
    from lib.distribution import *
    from lib.agent_memory import AgentMemory
    from lib.agent_graph import AgentGraph
except ImportError:
    from core import *
    from distribution import *
    from lib.agent_memory import AgentMemory
    from agent_graph import AgentGraph


@typechecked
def generate_position_with_distribution(graph: nx.Graph, num_nodes: int, dist_type: str, param, center_node: Optional[int] = None, debug: Optional[bool] = False) -> Tuple[list, Optional[int]]:
    """
    Picks a center node (provided or randomly selected) from the graph, then generates positions using the given distribution.

    Parameters:
    -----------
    graph : nx.Graph
        The input graph.
    num_nodes : int
        The number of nodes to select.
    dist_type : str
        The distribution type to use. Options include:
          - "uniform": Uses distribute_uniform_random (param is max_distance)
          - "normal": Uses distribute_normal (param should be a tuple (mean_distance, std_dev))
          - "exponential": Uses distribute_exponential (param is scale)
          - "power_law": Uses distribute_power_law (param is exponent)
          - "beta": Uses distribute_beta (param should be a tuple (alpha, beta))
          - "high_degree": Uses distribute_degree_weighted with favor_high_degree=True
          - "low_degree": Uses distribute_degree_weighted with favor_high_degree=False
    param : varies
        Parameter(s) required for the selected distribution.
    center_node : Optional[int]
        The center node id to use. If None, a random center node is selected.
    debug : Optional[bool]
        If True, debug messages will be printed during the process.

    Returns:
    --------
    tuple
        (positions, center_node), where positions is a list of selected node ids.
        Returns (None, None) in case of an error.
    """
    # If no center_node provided, choose one randomly
    if center_node is None:
        try:
            center_node = random.choice([n for n in graph.nodes() if isinstance(n, int)])
        except Exception as e:
            error(f"Error selecting random center node: {e}")
            return None, None
    else:
        # Verify that the provided center_node is in the graph
        if center_node not in graph.nodes():
            error(f"Provided center_node {center_node} is not in the graph.")
            return None, None

    if dist_type == "uniform":
        positions = distribute_uniform_random(graph, center_node, num_nodes, max_distance=param)
    elif dist_type == "normal":
        try:
            mean_d, std = param
        except Exception as e:
            error(f"Invalid parameter for normal distribution: {param}. Error: {e}")
            return None, center_node
        positions = distribute_normal(graph, center_node, num_nodes, mean_distance=mean_d, std_dev=std)
    elif dist_type == "exponential":
        positions = distribute_exponential(graph, center_node, num_nodes, scale=param)
    elif dist_type == "power_law":
        positions = distribute_power_law(graph, center_node, num_nodes, exponent=param)
    elif dist_type == "beta":
        try:
            alpha, beta_param = param
        except Exception as e:
            error(f"Invalid parameter for beta distribution: {param}. Error: {e}")
            return None, center_node
        positions = distribute_beta(graph, center_node, num_nodes, alpha=alpha, beta=beta_param)
    elif dist_type == "high_degree":
        positions = distribute_degree_weighted(graph, center_node, num_nodes, favor_high_degree=True)
    elif dist_type == "low_degree":
        positions = distribute_degree_weighted(graph, center_node, num_nodes, favor_high_degree=False)
    else:
        warning(f"Distribution type '{dist_type}' not recognized. Using default center positions.")
        positions = [center_node] * num_nodes
    info(f"Generated {num_nodes} positions using distribution: {dist_type}", debug)
    return positions, center_node


@typechecked
def read_yml_file(file_path: str) -> Dict:
    """
    Reads a YAML file from the given file path and returns its contents as a dictionary.

    Parameters:
    -----------
    file_path : str
        The path to the YAML file.

    Returns:
    --------
    Dict
        The configuration dictionary loaded from the YAML file.
    """
    try:
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        error(f"Error reading YAML file at {file_path}: {e}")
        raise Exception(f"Error reading YAML file at {file_path}: {e}")


@typechecked
def recursive_update(default: Dict, override: Dict, force: bool, debug: Optional[bool] = False) -> Dict:
    """
    Recursively updates the 'default' dictionary with the 'override' dictionary.

    For each key in the override dictionary:
      - If force is True:
          - If the key exists in default, override the value and print a warning.
          - If the key does not exist in default, add the key with the override value and print a debug message.
      - If force is False:
          - If the key exists in default and its value is None or "Error", override and print a debug message.
          - If the key does not exist in default, add it with the override value and print a debug message.

    If both values are dictionaries, the function updates them recursively.

    Parameters:
    -----------
    default : Dict
        The original configuration dictionary.
    override : Dict
        The extra (override) dictionary.
    force : bool
        Whether to force overriding keys that already have a valid value.

    Returns:
    --------
    Dict
        The updated dictionary.
    """
    for key, value in override.items():
        # If both default and override values are dictionaries, update recursively.
        if key in default and isinstance(default[key], dict) and isinstance(value, dict):
            default[key] = recursive_update(default[key], value, force)
        else:
            # Force is True: Always override or add.
            if force:
                if key in default:
                    # Check if the keys are the same
                    if default[key] != value:
                        warning(f"Overriding key '{key}': {default[key]} -> {value}", debug)
                        default[key] = value
                else:
                    info(f"Key '{key}' not found in original config. Adding with value: {value}", debug)
                    default[key] = value
            else:
                # Force is False: Only override if key is missing or its value is None or "Error".
                if key in default:
                    current = default.get(key)
                    if current is None or current == "Error":
                        info(f"Key '{key}' is missing or invalid (current: {current}). Setting to: {value}", debug)
                        default[key] = value
                else:
                    info(f"Key '{key}' not found in original config. Adding with value: {value}", debug)
                    default[key] = value
    return default


@typechecked
def write_yaml_config(config: Dict, output_dir: str, filename: str) -> bool:
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        warning(f"File {filepath} already exists. Skipping write.")
        return False
    try:
        with open(filepath, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
        return True
    except Exception as e:
        error(f"Error writing config to file {filepath}: {e}")
        return False


@typechecked
def load_config_metadata(config: Dict) -> Dict[str, Any]:
    metadata = {}
    # Graph file name
    metadata["graph_file"] = config["environment"]["graph_name"]

    # Flag parameters
    flag_config = config["extra_prameters"]["parameters"]["flag"]
    metadata["flag_num"] = flag_config["number"]
    metadata["flag_dist_type"] = flag_config["distribution"]["type"]
    metadata["flag_param"] = flag_config["distribution"]["param"]

    # Attacker parameters
    attacker_config = config["extra_prameters"]["parameters"]["attacker"]
    metadata["attacker_num"] = attacker_config["number"]
    metadata["attacker_dist_type"] = attacker_config["distribution"]["type"]
    metadata["attacker_param"] = attacker_config["distribution"]["param"]

    # Defender parameters
    defender_config = config["extra_prameters"]["parameters"]["defender"]
    metadata["defender_num"] = defender_config["number"]
    metadata["defender_dist_type"] = defender_config["distribution"]["type"]
    metadata["defender_param"] = defender_config["distribution"]["param"]

    return metadata


@typechecked
def generate_config_parameters(
    graph_file: str,
    game_rule: str,
    flag_num: int,
    flag_dist_type: str,
    flag_param: Any,
    center_node_flag: Any,
    flag_positions: Any,
    attacker_num: int,
    attacker_dist_type: str,
    attacker_param: Any,
    center_node_attacker: Any,
    attacker_positions: Any,
    defender_num: int,
    defender_dist_type: str,
    defender_param: Any,
    center_node_defender: Any,
    defender_positions: Any,
) -> Tuple[Dict, str]:
    # Build individual attacker and defender configurations
    ATTACKER_CONFIG = {f"attacker_{i}": {"start_node_id": attacker_positions[i]} for i in range(len(attacker_positions))}
    DEFENDER_CONFIG = {f"defender_{i}": {"start_node_id": defender_positions[i]} for i in range(len(defender_positions))}

    # Build the parameters information to be stored under extra_prameters
    parameters = {
        "flag": {
            "center_node": center_node_flag,
            "number": flag_num,
            "distribution": {"type": flag_dist_type, "param": flag_param},
        },
        "attacker": {
            "center_node": center_node_attacker,
            "number": attacker_num,
            "distribution": {"type": attacker_dist_type, "param": attacker_param},
        },
        "defender": {
            "center_node": center_node_defender,
            "number": defender_num,
            "distribution": {"type": defender_dist_type, "param": defender_param},
        },
    }

    # Get current date and time up to minutes
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3]

    # Create a unique hash based on the PARAMETERS information, other key parts, and the timestamp.
    config_str = str(parameters) + graph_file + game_rule + timestamp
    hash_key = hashlib.sha256(config_str.encode()).hexdigest()[:10]

    # Build the generated configuration (partial) from the parameters
    generated_config = {
        "game": {
            "rule": game_rule,
            # Note: Other game settings (max_time, interaction, payoff, etc.) will be filled by the default config.
            "flag": {
                "positions": flag_positions,
            },
        },
        "environment": {
            "graph_name": graph_file,
        },
        "agents": {
            "attacker_config": ATTACKER_CONFIG,
            "defender_config": DEFENDER_CONFIG,
        },
        # Store the original PARAMETERS info, the generated CONFIG_ID, and timestamp in extra_prameters
        "extra_prameters": {
            "parameters": parameters,
            "CONFIG_ID": hash_key,
            "timestamp": timestamp,
        },
    }
    return generated_config, hash_key


@typechecked
def generate_single_config(
    graph: nx.Graph,
    graph_file: str,
    flag_num: int,
    flag_dist_type: str,
    flag_param: Any,
    attacker_num: int,
    defender_num: int,
    attacker_dist_type: str,
    attacker_param: Any,
    defender_dist_type: str,
    defender_param: Any,
    game_rule: str,
    output_dir: str,
    default_config_path: str,  # New parameter for the default config file
    debug: Optional[bool] = False,
    center_node: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Generates a single configuration file based on the given parameters.
    It loads a default configuration from 'default_config_path' and fills in missing keys.

    Parameters:
    -----------
    graph : nx.Graph
        The graph object.
    graph_file : str
        The name of the graph file.
    flag_num : int
        Number of flags.
    flag_dist_type : str
        Distribution type for flag positions.
    flag_param : Any
        Parameter(s) for the flag distribution.
    attacker_num : int
        Number of attacker agents.
    defender_num : int
        Number of defender agents.
    attacker_dist_type : str
        Distribution type for attacker positions.
    attacker_param : Any
        Parameter(s) for the attacker distribution.
    defender_dist_type : str
        Distribution type for defender positions.
    defender_param : Any
        Parameter(s) for the defender distribution.
    game_rule : str
        The game rule to include in the configuration.
    output_dir : str
        Directory where the configuration file will be saved.
    default_config_path : str
        Path to the default configuration YAML file.
    debug : Optional[bool]
        If True, debug messages will be printed during the process.

    Returns:
    --------
    bool
        True if the configuration was generated successfully, False otherwise.
    """
    # Generate positions for flag, attacker, and defender using provided functions.
    # These functions should return positions and a center node.
    flag_positions, center_node_flag = generate_position_with_distribution(graph, flag_num, flag_dist_type, flag_param, center_node=center_node)
    if flag_positions is None:
        error(f"Flag position generation failed for graph {graph_file} with parameters: flag_num={flag_num}, distribution={flag_dist_type}, param={flag_param}")
        return False

    attacker_positions, center_node_attacker = generate_position_with_distribution(graph, attacker_num, attacker_dist_type, attacker_param, center_node=center_node)
    if attacker_positions is None:
        error(f"Attacker position generation failed for graph {graph_file} with parameters: attacker_num={attacker_num}, distribution={attacker_dist_type}, param={attacker_param}")
        return False

    defender_positions, center_node_defender = generate_position_with_distribution(graph, defender_num, defender_dist_type, defender_param, center_node=center_node)
    if defender_positions is None:
        error(f"Defender position generation failed for graph {graph_file} with parameters: defender_num={defender_num}, distribution={defender_dist_type}, param={defender_param}")
        return False

    # Build the generated configuration (partial) and compute CONFIG_ID
    generated_config, hash_key = generate_config_parameters(
        graph_file=graph_file,
        game_rule=game_rule,
        flag_num=flag_num,
        flag_dist_type=flag_dist_type,
        flag_param=flag_param,
        center_node_flag=center_node_flag,
        flag_positions=flag_positions,
        attacker_num=attacker_num,
        attacker_dist_type=attacker_dist_type,
        attacker_param=attacker_param,
        center_node_attacker=center_node_attacker,
        attacker_positions=attacker_positions,
        defender_num=defender_num,
        defender_dist_type=defender_dist_type,
        defender_param=defender_param,
        center_node_defender=center_node_defender,
        defender_positions=defender_positions,
    )

    # Load the default configuration
    try:
        default_config = read_yml_file(default_config_path)
    except Exception as e:
        error(e)
        return False, ""

    # Merge the generated configuration into the default config (generated values override defaults)
    merged_config = recursive_update(generated_config, default_config, debug=debug, force=False)

    # Write the merged configuration to a YAML file
    filename = f"config_{hash_key}.yml"
    if not write_yaml_config(merged_config, output_dir, filename):
        return False

    success(f"Generated configuration: {filename}", debug)
    return True, filename


@typechecked
def extract_positions_from_config(config: Dict[str, Any]) -> Tuple[List[int], List[int], List[int], Optional[str]]:
    """
    Extract the attacker and defender start node IDs, flag positions, and graph name from a configuration dictionary.

    Args:
        config (Dict[str, Any]): The configuration dictionary created by generate_config_parameters

    Returns:
        Tuple[List[int], List[int], List[int], Optional[str]]: A tuple containing:
            - List of attacker start node IDs
            - List of defender start node IDs
            - List of flag positions
            - Graph name (or None if not found)
    """
    # Extract attacker start node IDs
    attacker_config = config.get("agents", {}).get("attacker_config", {})
    attacker_positions = []
    for i in range(len(attacker_config)):
        # Try with both string and integer keys
        key = f"attacker_{i}"
        if key in attacker_config and "start_node_id" in attacker_config[key]:
            attacker_positions.append(attacker_config[key]["start_node_id"])

    # Extract defender start node IDs
    defender_config = config.get("agents", {}).get("defender_config", {})
    defender_positions = []
    for i in range(len(defender_config)):
        # Try with both string and integer keys
        key = f"defender_{i}"
        if key in defender_config and "start_node_id" in defender_config[key]:
            defender_positions.append(defender_config[key]["start_node_id"])

    # Extract flag positions
    flag_positions = config.get("game", {}).get("flag", {}).get("positions", [])

    # Extract graph name
    graph_name = config.get("environment", {}).get("graph_name")

    return attacker_positions, defender_positions, flag_positions, graph_name


@typechecked
def apply_game_rule_overrides(config: Dict, game_rule_path: str, debug: Optional[bool] = False) -> Dict:

    # Check if game rule is in the config.
    if "game" not in config or "rule" not in config["game"]:
        warning("No game rule found in the configuration. Skipping game rule overrides.")
        return config
    game_rule_name = config["game"]["rule"]
    game_rule_file = os.path.join(game_rule_path, f"{game_rule_name}.yml")
    try:
        gr = read_yml_file(game_rule_file).pop("gamerule", {})
    except Exception as e:
        error(f"Error reading game rule file {game_rule_file}: {e}")
        return config
    if not gr:
        warning(f"No gamerule found in {game_rule_file}. Skipping game rule overrides.")
        return config

    # Process non-agent keys first.
    for key, value in gr.items():
        if key != "agents":
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                # Use force=True to override all keys.
                config[key] = recursive_update(config[key], value, force=True)
            else:
                if key in config:
                    warning(f"Overriding key '{key}': {config[key]} -> {value}", debug)
                else:
                    info(f"Key '{key}' not found in original config. Adding with value: {value}", debug)
                config[key] = value

    # --- Override the agents section ---
    if "agents" in gr:
        agents_overrides = gr["agents"]

        # Process attacker overrides.
        if "attacker_global" in agents_overrides:
            attacker_override = agents_overrides["attacker_global"]
            if "agents" in config:
                # Override global attacker settings.
                if "attacker_global" in config["agents"]:
                    old_value = config["agents"]["attacker_global"]
                    new_value = attacker_override.copy()
                    if old_value != new_value:
                        warning(f"Overriding agents.attacker_global: {old_value} -> {new_value}", debug)
                        config["agents"]["attacker_global"] = new_value
                # Override each individual attacker.
                if "attacker_config" in config["agents"]:
                    for key, a_conf in config["agents"]["attacker_config"].items():
                        old_value = a_conf.copy()
                        start_node = a_conf.get("start_node_id")
                        new_conf = attacker_override.copy()
                        if start_node is not None:
                            new_conf["start_node_id"] = start_node
                        if old_value != new_conf:
                            warning(f"Overriding agents.attacker_config.{key}: {old_value} -> {new_conf}", debug)
                            config["agents"]["attacker_config"][key] = new_conf

        # Process defender overrides.
        if "defender_global" in agents_overrides:
            defender_override = agents_overrides["defender_global"]
            if "agents" in config:
                # Override global defender settings.
                if "defender_global" in config["agents"]:
                    old_value = config["agents"]["defender_global"]
                    new_value = defender_override.copy()
                    if old_value != new_value:
                        warning(f"Overriding agents.defender_global: {old_value} -> {new_value}", debug)
                        config["agents"]["defender_global"] = new_value
                # Override each individual defender.
                if "defender_config" in config["agents"]:
                    for key, d_conf in config["agents"]["defender_config"].items():
                        old_value = d_conf.copy()
                        start_node = d_conf.get("start_node_id")
                        new_conf = defender_override.copy()
                        if start_node is not None:
                            new_conf["start_node_id"] = start_node
                        if old_value != new_conf:
                            warning(f"Overriding agents.defender_config.{key}: {old_value} -> {new_conf}", debug)
                            config["agents"]["defender_config"][key] = new_conf
    return config


@typechecked
def get_directories(root_dir: str) -> dict:
    """
    Build and return a dictionary of common directories.
    """
    return {
        "config": os.path.join(root_dir, "data/config"),
        "graph": os.path.join(root_dir, "data/graphs"),
        "rules": os.path.join(root_dir, "data/rules"),
        "result": os.path.join(root_dir, "data/result"),
    }


@typechecked
def load_configuration(config_name: str, dirs: dict, debug: bool = False) -> dict:
    """
    Load and process the configuration file.
    """
    config_file = os.path.join(dirs["config"], config_name)
    original_config = read_yml_file(config_file)
    success("Read original config file successfully", debug)
    config = apply_game_rule_overrides(original_config, dirs["rules"])
    return config


@typechecked
def load_graph(config: dict, dirs: dict, debug: bool = False) -> nx.MultiDiGraph:
    """
    Load the game graph from file.
    """
    graph_name = config["environment"]["graph_name"]
    graph_path = os.path.join(dirs["graph"], graph_name)
    try:
        from lib.graph_utils import export_graph
    except ImportError:
        from graph_utils import export_graph
    G = export_graph(graph_path)
    if not isinstance(G, nx.MultiDiGraph):
        warning(f"Graph {graph_name} is not a MultiDiGraph!")
    success(f"Loaded graph: {graph_name}", debug)
    return G


@typechecked
def create_static_sensors() -> dict:
    """
    Create static sensor definitions that can be reused across contexts.
    """
    return {"map": gamms.sensor.SensorType.MAP, "agent": gamms.sensor.SensorType.AGENT, "neighbor": gamms.sensor.SensorType.NEIGHBOR}


@typechecked
def create_context_with_sensors(config: dict, G: nx.MultiDiGraph, visualization: bool, static_sensors: dict, debug: bool = False):
    """
    Create a new game context, attach the graph, and create the sensors using pre-initialized definitions.
    """
    # Choose visualization engine
    if not visualization:
        VIS_ENGINE = gamms.visual.Engine.NO_VIS
    else:
        if config["visualization"]["visualization_engine"] == "PYGAME":
            VIS_ENGINE = gamms.visual.Engine.PYGAME
        else:
            VIS_ENGINE = gamms.visual.Engine.NO_VIS
    # success(f"Visualization Engine: {VIS_ENGINE}", debug)

    # Create a new context
    ctx = gamms.create_context(vis_engine=VIS_ENGINE)
    ctx.graph.attach_networkx_graph(G)

    # Create sensors using static definitions
    for sensor_name, sensor_type in static_sensors.items():
        ctx.sensor.create_sensor(sensor_name, sensor_type)

    return ctx


# =====================================================================
# FUNCTIONS FOR GAME.PY
# =====================================================================


def initialize_agents(ctx: Any, config: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, AgentMemory]]:
    """
    Configure and create agents in the game context based on a structured config dict.
    Individual agent parameters override global defaults when specified.

    Args:
        ctx: Game context object with agent creation capabilities
        config: Configuration dictionary containing agent settings

    Returns:
        Tuple containing:
            - agent_config: Dictionary mapping agent names to their configuration settings
            - agent_params_dict: Dictionary mapping agent names to their AgentMemory objects
    """
    # Extract agent configurations
    attacker_config = config.get("agents", {}).get("attacker_config", {})
    defender_config = config.get("agents", {}).get("defender_config", {})

    # Extract global defaults
    attacker_global = config.get("agents", {}).get("attacker_global", {})
    defender_global = config.get("agents", {}).get("defender_global", {})

    # Extract visualization settings
    vis_settings = config.get("visualization", {})
    colors = vis_settings.get("colors", {})
    sizes = vis_settings.get("sizes", {})

    # Set default values if not provided
    global_agent_size = sizes.get("global_agent_size", 10)

    def get_agent_param(agent_config: Dict[str, Any], param_name: str, global_config: Dict[str, Any]) -> Any:
        """Get parameter with priority: individual config > global params"""
        return agent_config.get(param_name, global_config.get(param_name))

    def create_agent_entries(configs: Dict[str, Dict[str, Any]], team: str, global_config: Dict[str, Any], team_color: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, AgentMemory]]:
        """
        Create agent entries and memory objects for a team

        Args:
            configs: Dictionary of agent configurations
            team: Team name (attacker/defender)
            global_config: Global configuration for the team
            team_color: Default color for the team

        Returns:
            Tuple of (agent_entries, agent_memories)
        """
        entries: Dict[str, Dict[str, Any]] = {}
        memories: Dict[str, AgentMemory] = {}

        for name, config in configs.items():
            # Ensure config is a dictionary (handle empty configs)
            if config is None:
                config = {}

            start_node_id = config.get("start_node_id")
            if start_node_id is None:
                warning(f"{name} has no start_node_id. Skipping.")
                continue

            # Get parameters with fallback to global defaults
            speed = get_agent_param(config, "speed", global_config)
            capture_radius = get_agent_param(config, "capture_radius", global_config)
            sensors = get_agent_param(config, "sensors", global_config)
            color = colors.get(f"{team}_global", team_color)

            # Create agent entry for the context
            entries[name] = {"team": team, "sensors": sensors, "color": color, "current_node_id": start_node_id, "start_node_id": start_node_id, "size": global_agent_size}

            # Extract known parameters
            known_params = ["speed", "capture_radius", "sensors", "start_node_id"]

            # Get any extra parameters as kwargs
            extra_params = {k: v for k, v in config.items() if k not in known_params}

            # Create parameter object with both required and extra parameters
            memories[name] = AgentMemory(speed=speed, capture_radius=capture_radius, map=AgentGraph(), start_node_id=start_node_id, **extra_params)

        # success(f"Created {len(entries)} {team} agents.")
        return entries, memories

    # Default colors if not specified
    default_attacker_color = "red"
    default_defender_color = "blue"

    # Create entries for both teams
    attacker_entries, attacker_memories = create_agent_entries(attacker_config, "attacker", attacker_global, default_attacker_color)

    defender_entries, defender_memories = create_agent_entries(defender_config, "defender", defender_global, default_defender_color)

    # Combine configurations
    agent_config = {**attacker_entries, **defender_entries}
    agent_params_dict = {**attacker_memories, **defender_memories}

    # Create agents in context
    for name, config in agent_config.items():
        ctx.agent.create_agent(name, **config)

    success(f"Created {len(attacker_entries)} attackers and {len(defender_memories)} defenders.")

    return agent_config, agent_params_dict


def assign_strategies(ctx: Any, agent_config: Dict[str, Dict[str, Any]], attacker_strategy_module: Any, defender_strategy_module: Any) -> None:
    """
    Assign strategies to agents based on their team.

    This function maps agent configurations to strategies using separate strategy modules
    for attackers and defenders. Then, it registers the strategy with each agent in the context.

    Args:
        ctx (Any): The initialized game context with agent management.
        agent_config (Dict[str, Dict[str, Any]]): Dictionary of agent configurations keyed by agent name.
        attacker_strategy_module (Any): Module providing strategies for attackers via a `map_strategy` function.
        defender_strategy_module (Any): Module providing strategies for defenders via a `map_strategy` function.

    Returns:
        None
    """
    try:
        strategies: Dict[str, Any] = {}
        # Build strategy mappings for attackers and defenders.
        attacker_configs = {name: config for name, config in agent_config.items() if config.get("team") == "attacker"}
        defender_configs = {name: config for name, config in agent_config.items() if config.get("team") == "defender"}

        strategies.update(attacker_strategy_module.map_strategy(attacker_configs))
        strategies.update(defender_strategy_module.map_strategy(defender_configs))

        # Register each agent's strategy if available.
        for agent in ctx.agent.create_iter():
            agent.register_strategy(strategies.get(agent.name))

        # success("Strategies assigned to agents.")
    except Exception as e:
        error(f"Error assigning strategies: {e}")


def configure_visualization(ctx: Any, agent_config: Dict[str, Dict[str, Any]], config: Dict[str, Any]) -> None:
    """
    Configure visualization settings for the game graph and agents.

    This function extracts visualization parameters from the config dictionary,
    sets up the global visualization parameters for the graph, and configures
    individual visualization parameters for each agent.

    Args:
        ctx (Any): The initialized game context that contains visualization methods.
        agent_config (Dict[str, Dict[str, Any]]): Dictionary of agent configurations keyed by agent name.
        config (Dict[str, Any]): Complete configuration dictionary containing visualization settings.

    Returns:
        None
    """
    # Extract visualization settings from config
    vis_config = config.get("visualization", {})

    # Extract window size with defaults
    window_size = vis_config.get("window_size", [1980, 1080])
    width = window_size[0] if isinstance(window_size, list) and len(window_size) > 0 else 1980
    height = window_size[1] if isinstance(window_size, list) and len(window_size) > 1 else 1080

    # Extract other visualization parameters with defaults
    draw_node_id = vis_config.get("draw_node_id", False)
    game_speed = vis_config.get("game_speed", 1)

    # Get color settings
    colors = vis_config.get("colors", {})

    # Import color constants if they're being used
    try:
        from gamms.VisualizationEngine import Color

        node_color = Color.Black
        edge_color = Color.Gray
        default_color = Color.White
    except ImportError:
        # Fallback to string colors if Color class isn't available
        node_color = "black"
        edge_color = "gray"
        default_color = "white"

    # Get size settings
    sizes = vis_config.get("sizes", {})
    default_size = sizes.get("global_agent_size", 10)

    # Set global graph visualization parameters
    ctx.visual.set_graph_visual(width=width, height=height, draw_id=draw_node_id, node_color=node_color, edge_color=edge_color)

    # Set game speed
    ctx.visual._sim_time_constant = game_speed

    # Set individual agent visualization parameters
    for name, agent_cfg in agent_config.items():
        # Determine agent's team to get the right color
        team = agent_cfg.get("team", "")
        team_color = colors.get(f"{team}_global", default_color)

        # Get color and size with appropriate defaults
        color = agent_cfg.get("color", team_color)
        size = agent_cfg.get("size", default_size)

        # Apply visual settings to the agent
        ctx.visual.set_agent_visual(name, color=color, size=size)

    success("Visualization configured.")


def initialize_flags(ctx: Any, config: Dict[str, Any], debug: Optional[bool] = False) -> None:
    """
    Initializes flags in the simulation based on configuration settings.

    Extracts flag positions and visual properties from the config dictionary,
    then creates flag visual elements at the specified node locations.

    Args:
        ctx (Any): The initialized game context with graph and visual components.
        config (Dict[str, Any]): Complete configuration dictionary.
        debug (Optional[bool]): If True, debug messages will be printed during the process.

    Returns:
        None
    """

    # Extract flag positions from config
    flag_positions = config.get("game", {}).get("flag", {}).get("positions", [])

    if not flag_positions:
        warning("No flag positions found in config.")
        return

    # Extract visualization settings
    vis_config = config.get("visualization", {})
    colors = vis_config.get("colors", {})
    sizes = vis_config.get("sizes", {})

    # Get flag color and size with defaults
    flag_color = colors.get("flag", "green")
    flag_size = sizes.get("flag_size", 10)

    # Try to use Color constants if available
    try:
        from gamms.VisualizationEngine import Color

        color_map = {"green": Color.Green, "red": Color.Red, "blue": Color.Blue, "yellow": Color.Yellow, "white": Color.White, "black": Color.Black, "gray": Color.Gray}
        # Convert string color to Color constant if possible
        if isinstance(flag_color, str) and flag_color.lower() in color_map:
            flag_color = color_map[flag_color.lower()]
    except (ImportError, AttributeError):
        # Keep using string color names if Color class isn't available
        pass

    # Create flags for each position
    for index, flag_node_id in enumerate(flag_positions):
        try:
            # Get node coordinates from the graph
            node = ctx.graph.graph.get_node(flag_node_id)

            # Create flag visual data
            flag_data = {"x": node.x, "y": node.y, "scale": flag_size, "color": flag_color}

            # Add the flag as a visual element
            ctx.visual.add_artist(f"flag_{index}", flag_data)
            info(f"Flag {index} created at node {flag_node_id}", debug)

        except Exception as e:
            error(f"Error creating flag at node {flag_node_id}: {str(e)}")
            raise e

    success(f"Initialized {len(flag_positions)} flags.")


def handle_interaction(ctx: Any, agent: Any, action: str, processed: Set[str], agent_params: Dict[str, Any]) -> bool:
    """
    Handle agent interaction (kill or respawn).

    Args:
        ctx (Any): The context containing agent management methods.
        agent (Any): The agent instance.
        action (str): The action to perform ("kill" or "respawn").
        processed (Set[str]): Set of agent names already processed.
        agent_params (Dict[str, Any]): Mapping from agent name to its parameters.

    Returns:
        bool: True if an interaction was processed, False otherwise.
    """
    if action == "kill":
        ctx.agent.delete_agent(agent.name)
        processed.add(agent.name)
        return True
    if action == "respawn":
        agent.prev_node_id = agent.current_node_id
        # Use start position from agent_params
        start_pos = agent_params[agent.name].start_node_id
        agent.current_node_id = start_pos
        return True
    return False


def check_agent_interaction(
    ctx: Any, G: nx.Graph, agent_params: Dict[str, Any], flag_positions: List[Any], interaction_config: Dict[str, Any], time: float, debug: Optional[bool] = True
) -> Tuple[int, int, int, int, List[Tuple[str, Any]], List[Tuple[str, str]]]:
    """
    Main interaction checking function between agents and flags.

    Args:
        ctx (Any): The context with agent management.
        G (nx.Graph): The graph representing the game map.
        agent_params (Dict[str, Any]): Mapping of agent names to their parameters.
        flag_positions (List[Any]): List of nodes where flags are located.
        interaction_config (Dict[str, Any]): Dictionary containing configuration parameters for interactions.
        time (float): The current simulation time.
        debug (Optional[bool]): If True, debug messages will be printed during the process.

    Returns:
        Tuple containing:
            - captures (int): Number of captures.
            - tags (int): Number of tags.
            - remaining_attackers (int): Remaining attackers.
            - remaining_defenders (int): Remaining defenders.
            - capture_details (List[Tuple[str, Any]]): List of tuples (attacker_name, flag) indicating captures.
            - tagging_details (List[Tuple[str, str]]): List of tuples (defender_name, attacker_name) indicating tags.
    """
    captures = tags = 0
    processed: Set[str] = set()
    capture_details: List[Tuple[str, Any]] = []
    tagging_details: List[Tuple[str, str]] = []

    # Get initial lists of agents by team
    attackers = [a for a in ctx.agent.create_iter() if a.team == "attacker"]
    defenders = [d for d in ctx.agent.create_iter() if d.team == "defender"]

    # Process interactions based on priority
    if interaction_config["prioritize"] == "capture":
        # Check flag captures first
        for attacker in attackers[:]:
            if attacker.name in processed:
                continue
            for flag in flag_positions:
                shortest_distance = nx.shortest_path_length(G, attacker.current_node_id, flag)
                attacker_capture_radius = getattr(agent_params[attacker.name], "capture_radius", 0)
                if shortest_distance <= attacker_capture_radius:
                    info(f"Attacker {attacker.name} captured flag {flag} at time {time}", debug)
                    if handle_interaction(ctx, attacker, interaction_config["capture"], processed, agent_params):
                        captures += 1
                        capture_details.append((attacker.name, flag))
                        break
    # Check combat interactions
    for defender in defenders[:]:
        if defender.name in processed:
            continue
        for attacker in attackers[:]:
            if attacker.name in processed or defender.name in processed:
                continue
            try:
                defender_capture_radius = getattr(agent_params[defender.name], "capture_radius", 0)
                if nx.shortest_path_length(G, attacker.current_node_id, defender.current_node_id) <= defender_capture_radius:
                    info(f"Defender {defender.name} tagged attacker {attacker.name} at time {time}", debug)
                    if interaction_config["tagging"] == "both_kill":
                        handle_interaction(ctx, attacker, "kill", processed, agent_params)
                        handle_interaction(ctx, defender, "kill", processed, agent_params)
                    elif interaction_config["tagging"] == "both_respawn":
                        handle_interaction(ctx, attacker, "respawn", processed, agent_params)
                        handle_interaction(ctx, defender, "respawn", processed, agent_params)
                    else:
                        handle_interaction(ctx, attacker, interaction_config["tagging"], processed, agent_params)
                    tags += 1
                    tagging_details.append((defender.name, attacker.name))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    # If tags processed first, check captures second
    if interaction_config["prioritize"] != "capture":
        for attacker in attackers[:]:
            if attacker.name in processed:
                continue
            for flag in flag_positions:
                shortest_distance = nx.shortest_path_length(G, attacker.current_node_id, flag)
                attacker_capture_radius = getattr(agent_params[attacker.name], "capture_radius", 0)
                if shortest_distance <= attacker_capture_radius:
                    info(f"Attacker {attacker.name} captured flag {flag} at time {time}", debug)
                    if handle_interaction(ctx, attacker, interaction_config["capture"], processed, agent_params):
                        captures += 1
                        capture_details.append((attacker.name, flag))

    # Count remaining agents
    remaining_attackers = sum(1 for a in ctx.agent.create_iter() if a.team == "attacker")
    remaining_defenders = sum(1 for d in ctx.agent.create_iter() if d.team == "defender")

    return captures, tags, remaining_attackers, remaining_defenders, capture_details, tagging_details


def check_termination(time: int, MAX_TIME: int, remaining_attackers: int, remaining_defenders: int) -> bool:
    """
    Check if the game should be terminated based on time or if one team is eliminated.

    Args:
        time (int): The current time step.
        MAX_TIME (int): The maximum allowed time steps.
        remaining_attackers (int): The number of remaining attackers.
        remaining_defenders (int): The number of remaining defenders.

    Returns:
        bool: True if termination condition is met, False otherwise.
    """
    if time >= MAX_TIME:
        success("Maximum time reached.")
        return True
    if remaining_attackers == 0:
        success("All attackers have been eliminated.")
        return True
    if remaining_defenders == 0:
        success("All defenders have been eliminated.")
        return True
    return False


def check_agent_dynamics(state: Dict[str, Any], agent_params: Any, G: nx.Graph) -> None:
    """
    Checks and adjusts the next node for an agent based on its speed and connectivity.

    Args:
        state (Dict[str, Any]): A dictionary containing the agent's current state with keys 'action', 'curr_pos', and 'name'.
        agent_params (Any): The agent's parameters including speed.
        G (nx.Graph): The graph representing the game environment.
    """
    agent_next_node = state["action"]
    agent_speed = agent_params.speed
    agent_prev_node = state["curr_pos"]
    if agent_next_node is None:
        agent_next_node = agent_prev_node
        warning(f"Agent {state['name']} has no next node, staying at {agent_prev_node}")
    try:
        shortest_path_length = nx.shortest_path_length(G, source=agent_prev_node, target=agent_next_node)
        if shortest_path_length > agent_speed:
            warning(f"Agent {state['name']} cannot reach {agent_next_node} from {agent_prev_node} within speed limit of {agent_speed}. Staying at {agent_prev_node}")
            state["action"] = agent_prev_node
    except nx.NetworkXNoPath:
        warning(f"No path from {agent_prev_node} to {agent_next_node}. Staying at {agent_prev_node}")
        state["action"] = agent_prev_node


def compute_payoff(payoff_config: Dict[str, Any], captures: int, tags: int) -> float:
    """
    Computes the payoff based on the specified model in the config.

    Args:
        payoff_config (Dict[str, Any]): Payoff configuration containing model name and constants
        captures (int): Number of attackers captured by defenders
        tags (int): Number of successful flag tags by attackers

    Returns:
        float: Calculated payoff value
    """
    # Check which payoff model to use
    model = payoff_config.get("model", "V1")

    if model == "V1":
        return V1(payoff_config, captures, tags)
    else:
        # Fallback to V1 if model not recognized
        return V1(payoff_config, captures, tags)


def V1(payoff_config: Dict[str, Any], captures: int, tags: int) -> float:
    """
    Original V1 payoff function: captures - k * tags

    Args:
        payoff_config (Dict[str, Any]): Payoff configuration containing constants
        captures (int): Number of attackers captured by defenders
        tags (int): Number of successful flag tags by attackers

    Returns:
        float: Calculated payoff value
    """
    # Extract k value from constants, default to 1.0 if not found
    constants = payoff_config.get("constants", {})
    k = constants.get("k", 1.0)

    # Calculate payoff
    payoff = captures - k * tags
    return payoff


def check_and_install_dependencies() -> bool:
    """
    Check if required packages are installed and install them if they're missing.

    Returns:
        bool: True if all dependencies are satisfied, False if installation failed.
    """
    import subprocess
    import sys

    # Required packages mapping: import_name -> pip package name
    required_packages = {
        "yaml": "pyyaml",
        "osmnx": "osmnx",
        "networkx": "networkx",
    }

    missing_packages: List[str] = []

    for import_name, pip_name in required_packages.items():
        try:
            __import__(import_name)
            success(f"✓ {import_name} is already installed")
        except ImportError:
            warning(f"✗ {import_name} is not installed")
            missing_packages.append(pip_name)

    if missing_packages:
        info("Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            for package in missing_packages:
                info(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                success(f"✓ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            error(f"Failed to install packages: {e}")
            warning("Please try installing the packages manually:\n" + "\n".join([f"pip install {pkg}" for pkg in missing_packages]))
            return False
        except Exception as e:
            error(f"An unexpected error occurred: {e}")
            return False

    success("All required dependencies are satisfied!")
    return True


def extract_map_sensor_data(state: dict) -> tuple:
    """
    Extracts the map sensor data from the given state.

    The map sensor data is expected to be a tuple containing:
        (sensor_type, (nodes_data, edges_data))

    Args:
        state (dict): The current state containing sensor data.

    Returns:
        tuple: A tuple (nodes_data, edges_data) representing the map data.

    Raises:
        ValueError: If map sensor data is not found or not in the expected format.
    """
    sensor_data = state.get("sensor", {})
    map_sensor = sensor_data.get("map")

    if map_sensor is None:
        raise ValueError("No map sensor data found in state.")

    # Unpack sensor data: expecting (sensor_type, map_data)
    sensor_type, map_data = map_sensor
    if not (isinstance(map_data, tuple) and len(map_data) == 2):
        raise ValueError("Map sensor data is not in the expected format (nodes_data, edges_data).")

    nodes_data, edges_data = map_data
    return nodes_data, edges_data


def extract_neighbor_sensor_data(state: dict) -> any:
    """
    Extracts the neighbor sensor data from the given state.

    The neighbor sensor data is expected to be a tuple containing:
        (sensor_type, neighbor_data)

    Args:
        state (dict): The current state containing sensor data.

    Returns:
        any: The neighbor sensor data.

    Raises:
        ValueError: If neighbor sensor data is not found in state.
    """
    sensor_data = state.get("sensor", {})
    neighbor_sensor = sensor_data.get("neighbor")

    if neighbor_sensor is None:
        raise ValueError("No neighbor sensor data found in state.")

    # Unpack the neighbor sensor tuple.
    sensor_type, neighbor_data = neighbor_sensor
    return neighbor_data


def extract_agent_sensor_data(state: dict) -> any:
    """
    Extracts the agent sensor data from the given state.

    The agent sensor data is expected to be a tuple containing:
        (sensor_type, agent_info)

    Args:
        state (dict): The current state containing sensor data.

    Returns:
        any: The agent sensor data.

    Raises:
        ValueError: If agent sensor data is not found in state.
    """
    sensor_data = state.get("sensor", {})
    agent_sensor = sensor_data.get("agent")

    if agent_sensor is None:
        raise ValueError("No agent sensor data found in state.")

    # Unpack the sensor tuple: (sensor_type, agent_info)
    sensor_type, agent_info = agent_sensor
    return agent_info


def initlize_agent_map(node_data: list, edge_data: list, agent_params: any, agent_info: any, flag_pos: list, flag_weight: Optional[list] = None) -> None:
    """
    Initialize the map with the given node and edge data.

    Args:
        node_data (list): A list of node data.
        edge_data (list): A list of edge data.
        agent_params (object): An object that includes a 'map' attribute, which supports
                               update_networkx_graph.

    Returns:
        None
    """
    agent_params.map.update_networkx_graph(node_data, edge_data)
    agent_params.map.set_flag_positions(flag_pos)
    agent_params.map.set_flag_weights(flag_weight)
    agent_params.map.set_agent_dict(agent_info)


def extract_sensor_data(state: dict, flag_pos: list, flag_weight: list, agent_params: any) -> tuple:
    """
    Extracts and processes sensor data from the state, updating the agent's map and returning
    the positions of attackers and defenders.

    This function performs the following steps:
        1. Extract map sensor data (nodes and edges) and update the agent's map.
        2. Extract agent sensor data and update agent positions on the map.
        3. Set flag positions and flag weights on the map.
        4. Retrieve positions for attackers and defenders.

    Args:
        state (dict): The current state containing sensor data.
        flag_pos (list): A list of flag positions.
        flag_weight (list): A list of flag weights.
        agent_params (object): An object that includes a 'map' attribute, which supports
                               update_networkx_graph, set_agent_positions, set_flag_positions,
                               set_flag_weights, and get_team_positions.

    Returns:
        tuple: A tuple (attacker_positions, defender_positions) representing the positions of agents
               on the map.
    """
    # Extract map data and update the graph.
    nodes_data, edges_data = extract_map_sensor_data(state)
    # Extract agent sensor data.
    agent_info = extract_agent_sensor_data(state)
    initlize_agent_map(nodes_data, edges_data, agent_params, agent_info, flag_pos, flag_weight)
    # Retrieve and return positions for attackers and defenders.
    attacker_positions = agent_params.map.get_team_positions("attacker")
    defender_positions = agent_params.map.get_team_positions("defender")

    return attacker_positions, defender_positions
